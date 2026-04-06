from utils import compute_ev_df
from config import game_id, token

import aiohttp
import asyncio
import sys
import os
import math
import logging
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.abspath("trading-simulator-client"))
from trading_client import *


# =========================
# GLOBAL PARAMS
# =========================

ALPHA = 0.80
DELTA = 2.0
K_INV = 2.5
MAX_POSITION = 60

EDGE_THRESHOLD = 1.0
EDGE_PCT_THRESHOLD = 0.10

BASE_SIZE = 20
EDGE_SCALE = 5.0

TICK_SIZE = 0.01
SLEEP_SECONDS = 30

# disagreement / confidence controls
MAX_DISAGREE_SCORE = 5.0
DISAGREE_INCREMENT = 1.0
DISAGREE_DECAY = 0.5
MIN_CONFIDENCE = 0.25
CONFIDENCE_SLOPE = 0.15


# =========================
# LOGGING
# =========================

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(handler)

# Uncomment for debugging if needed
# logging.disable(logging.CRITICAL)


# =========================
# HELPERS
# =========================

def round_to_tick(px: float, tick_size: float = TICK_SIZE, side=None) -> float:
    if pd.isna(px):
        return px

    if side == "BUY":
        return math.floor(px / tick_size) * tick_size
    elif side == "SELL":
        return math.ceil(px / tick_size) * tick_size
    else:
        return round(px / tick_size) * tick_size


def mid_price_map(order_books, fair_values):
    asks, bids, out = {}, {}, {}
    for team in fair_values:
        book = order_books[team]
        asks[team] = book.best_ask_px
        bids[team] = book.best_bid_px
        out[team] = (book.best_bid_px + book.best_ask_px) / 2
    return asks, bids, out


def thesis_disagreement(raw_edge: float, price_change: float) -> bool:
    """
    Disagreement means the market is moving farther away from your model value.

    If raw_edge > 0, your model says market is too cheap.
    If price falls further, disagreement increases.

    If raw_edge < 0, your model says market is too expensive.
    If price rises further, disagreement increases.
    """
    if pd.isna(price_change) or raw_edge == 0:
        return False
    return (raw_edge > 0 and price_change < 0) or (raw_edge < 0 and price_change > 0)


def score_to_confidence(score: float) -> float:
    """
    score 0 -> confidence 1.0
    score 5 -> confidence bottoms near MIN_CONFIDENCE
    """
    return max(MIN_CONFIDENCE, 1.0 - CONFIDENCE_SLOPE * score)


def compute_quote_sizes(edge: float, position: int) -> tuple[int, int]:
    """
    Returns (bid_size, ask_size), using effective edge.
    """
    edge_strength = min(abs(edge) / EDGE_SCALE, 1.0)
    base = BASE_SIZE * edge_strength

    pos_ratio = min(abs(position) / MAX_POSITION, 1.0)

    bid_size = 0.0
    ask_size = 0.0

    if edge > 0:
        # Model likes being longer, but reduce buying as inventory grows
        bid_size = base * (1.0 - pos_ratio)
        ask_size = base * (0.5 + 0.5 * pos_ratio)
    elif edge < 0:
        # Model likes being shorter, but reduce selling as inventory grows
        bid_size = base * (0.5 + 0.5 * pos_ratio)
        ask_size = base * (1.0 - pos_ratio)

    if position >= MAX_POSITION:
        bid_size = 0.0
    if position <= -MAX_POSITION:
        ask_size = 0.0

    return int(bid_size), int(ask_size)


def build_quotes(row):
    """
    Build quotes using effective fair value, not raw model fair value.
    """
    fair_value = row["effective_fair_value"]
    market_price = row["market_price"]
    position = row["position"]
    best_ask = row["best_ask"]
    best_bid = row["best_bid"]

    # quote center
    mid_quote = fair_value + ALPHA * (market_price - fair_value)

    # inventory skew
    pos_ratio = position / MAX_POSITION
    skew = K_INV * pos_ratio

    bid_quote = mid_quote - DELTA / 2 - skew
    ask_quote = mid_quote + DELTA / 2 - skew

    # softer asymmetry
    if position > 0:
        bid_quote -= 0.3 * (DELTA / 2)
        ask_quote -= 0.1 * (DELTA / 2)
    elif position < 0:
        bid_quote += 0.1 * (DELTA / 2)
        ask_quote += 0.3 * (DELTA / 2)

    bid_quote = round_to_tick(bid_quote, TICK_SIZE, side="BUY")
    ask_quote = round_to_tick(ask_quote, TICK_SIZE, side="SELL")

    # remain passive
    ask_quote = max(ask_quote, best_bid + TICK_SIZE)
    bid_quote = min(bid_quote, best_ask - TICK_SIZE)

    return pd.Series({
        "mid_quote": mid_quote,
        "bid_quote": bid_quote,
        "ask_quote": ask_quote,
    })


# =========================
# BOT
# =========================

class TradingBot(Client):
    def __init__(
        self,
        session: aiohttp.ClientSession,
        game_id: int,
        token: str,
    ) -> None:
        super().__init__(session, game_id, token)

    async def on_start(self) -> None:
        prev_prices = {}
        disagree_score = {}

        while True:
            order_books = await self.get_order_books()
            positions = self.positions

            # -------- fair values and market prices --------
            fair_value_df = compute_ev_df()
            fair_values = dict(zip(fair_value_df["team"], fair_value_df["EV"]))
            asks, bids, market_prices = mid_price_map(order_books, fair_values)

            common_teams = sorted(set(fair_values) & set(market_prices))

            df = pd.DataFrame({"team": common_teams})
            df["model_fair_value"] = df["team"].map(fair_values)
            df["best_ask"] = df["team"].map(asks)
            df["best_bid"] = df["team"].map(bids)
            df["market_price"] = df["team"].map(market_prices)
            df["position"] = df["team"].map(lambda t: positions.get(t, 0))

            # -------- raw model edge --------
            df["raw_edge"] = df["model_fair_value"] - df["market_price"]
            df["raw_edge_pct"] = df["raw_edge"] / df["market_price"]

            # -------- price change --------
            df["prev_price"] = df["team"].map(lambda t: prev_prices.get(t, pd.NA))
            df["price_change"] = df["market_price"] - df["prev_price"]

            # -------- disagreement score update --------
            for _, row in df.iterrows():
                team = row["team"]
                old_score = disagree_score.get(team, 0.0)

                if thesis_disagreement(row["raw_edge"], row["price_change"]):
                    new_score = min(old_score + DISAGREE_INCREMENT, MAX_DISAGREE_SCORE)
                else:
                    new_score = max(old_score - DISAGREE_DECAY, 0.0)

                disagree_score[team] = new_score

            df["disagree_score"] = df["team"].map(lambda t: disagree_score.get(t, 0.0))
            df["confidence"] = df["disagree_score"].apply(score_to_confidence)

            # -------- effective fair value / effective edge --------
            df["effective_fair_value"] = (
                df["market_price"] +
                df["confidence"] * (df["model_fair_value"] - df["market_price"])
            )
            df["effective_edge"] = df["effective_fair_value"] - df["market_price"]
            df["effective_edge_pct"] = df["effective_edge"] / df["market_price"]

            # -------- trade filter --------
            df["trade_candidate"] = (
                (df["effective_edge"].abs() > EDGE_THRESHOLD) |
                (df["effective_edge_pct"].abs() > EDGE_PCT_THRESHOLD)
            )

            df = df[df["trade_candidate"]].copy()

            # -------- quotes --------
            quote_df = df.apply(build_quotes, axis=1)
            df = pd.concat([df, quote_df], axis=1)

            # -------- sizes --------
            df[["bid_size", "ask_size"]] = df.apply(
                lambda row: pd.Series(
                    compute_quote_sizes(
                        edge=row["effective_edge"],
                        position=row["position"],
                    )
                ),
                axis=1,
            )

            # -------- update price memory --------
            for team, px in zip(df["team"], df["market_price"]):
                prev_prices[team] = px

            # -------- cancel all open orders every iteration --------
            logger.info("-------- CANCELING ALL OPEN ORDERS --------")
            current_open_orders = await self.get_open_orders()
            if current_open_orders:
                try:
                    await self.cancel_orders(list(current_open_orders.keys()))
                except Exception as e:
                    logger.info(f"Failed to cancel open orders: {e}")

            # -------- place fresh quotes --------
            logger.info("-------- SENDING NEW ORDERS --------")
            order_type = "LIMIT"

            for _, row in df.iterrows():
                team = row["team"]

                bid_quote = row["bid_quote"]
                ask_quote = row["ask_quote"]
                bid_size = int(row["bid_size"])
                ask_size = int(row["ask_size"])

                logger.info(
                    f"{team:12s} | "
                    f"ModelFV: {row['model_fair_value']:.2f} | "
                    f"EffFV: {row['effective_fair_value']:.2f} | "
                    f"Mkt: {row['market_price']:.2f} | "
                    f"RawEdge: {row['raw_edge']:.2f} | "
                    f"EffEdge: {row['effective_edge']:.2f} | "
                    f"Conf: {row['confidence']:.2f} | "
                    f"Pos: {row['position']:3d} | "
                    f"BidPx: {bid_quote:.2f} x {bid_size:2d} | "
                    f"AskPx: {ask_quote:.2f} x {ask_size:2d}"
                )

                if bid_size > 0:
                    try:
                        await self.send_order(team, bid_quote, bid_size, order_type)
                    except Exception as e:
                        logger.info(f"Failed to place BUY for {team}: {e}")

                if ask_size > 0:
                    try:
                        await self.send_order(team, ask_quote, -ask_size, order_type)
                    except Exception as e:
                        logger.info(f"Failed to place SELL for {team}: {e}")

            await asyncio.sleep(SLEEP_SECONDS)


async def main():
    async with create_session() as session:
        client = TradingBot(session, game_id, token)
        print(f"Access web view at {client.web_url}")
        await client.start()


if __name__ == "__main__":
    asyncio.run(main())