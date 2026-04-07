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
K_INV = 1.5
MAX_POSITION = 60
MIN_DELTA = 0.30
MAX_DELTA = 1.20
DELTA_MULTIPLIER = 0.7

EDGE_THRESHOLD = 0.6
EDGE_PCT_THRESHOLD = 0.10

BASE_SIZE = 30
EDGE_SCALE = 5.0

TICK_SIZE = 0.01
SLEEP_SECONDS = 10

# disagreement / confidence controls
MAX_DISAGREE_SCORE = 5.0
DISAGREE_INCREMENT = 1.0
DISAGREE_DECAY = 0.5
MIN_CONFIDENCE = 0.25
CONFIDENCE_SLOPE = 0.15

# inventory exit controls
INVENTORY_EXIT_THRESHOLD = 1
MIN_EXIT_SIZE = 1

# order update controls
PRICE_REPLACE_THRESHOLD = 0.10
SIZE_REPLACE_THRESHOLD = 0

# diagnostics
TRACE_ENABLED = True


# =========================
# LOGGING
# =========================

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(handler)

logger.propagate = False


# =========================
# HELPERS
# =========================

def fmt(x, ndigits=2):
    if pd.isna(x):
        return "NA"
    return f"{x:.{ndigits}f}"


def round_to_tick(px: float, tick_size: float = TICK_SIZE, side=None) -> float:
    """
    Round a price to the exchange tick.

    side:
        - 'BUY'  -> round down
        - 'SELL' -> round up
        - None   -> round to nearest
    """
    if pd.isna(px):
        return px

    if side == "BUY":
        return math.floor(px / tick_size) * tick_size
    elif side == "SELL":
        return math.ceil(px / tick_size) * tick_size
    else:
        return round(px / tick_size) * tick_size


def mid_price_map(order_books, fair_values):
    """
    Compute best ask, best bid, and midpoint for relevant teams.
    """
    asks, bids, mids = {}, {}, {}
    for team in fair_values:
        book = order_books[team]
        asks[team] = book.best_ask_px
        bids[team] = book.best_bid_px
        mids[team] = (book.best_bid_px + book.best_ask_px) / 2
    return asks, bids, mids


def thesis_disagreement(raw_edge: float, price_change: float) -> bool:
    """
    Disagreement means the market moved farther away from your model value.

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
    Returns (bid_size, ask_size) using effective edge.

    Risk-increasing side shrinks faster via squared decay.
    """
    edge_strength = min(abs(edge) / EDGE_SCALE, 1.0)
    base = BASE_SIZE * edge_strength

    pos_ratio = min(abs(position) / MAX_POSITION, 1.0)

    bid_size = 0.0
    ask_size = 0.0

    if edge > 0:
        # model wants to be longer
        bid_size = base * (1.0 - pos_ratio) ** 2
        ask_size = base * (0.5 + 0.5 * pos_ratio)

    elif edge < 0:
        # model wants to be shorter
        bid_size = base * (0.5 + 0.5 * pos_ratio)
        ask_size = base * (1.0 - pos_ratio) ** 2

    # hard caps
    if position >= MAX_POSITION:
        bid_size = 0.0
    if position <= -MAX_POSITION:
        ask_size = 0.0

    return int(bid_size), int(ask_size)


def build_quotes(row):
    fair_value = row["effective_fair_value"]
    market_price = row["market_price"]
    position = row["position"]
    best_ask = row["best_ask"]
    best_bid = row["best_bid"]

    spread = best_ask - best_bid
    delta_dynamic = max(MIN_DELTA, min(MAX_DELTA, DELTA_MULTIPLIER * spread))

    mid_quote = fair_value + ALPHA * (market_price - fair_value)

    pos_ratio = position / MAX_POSITION
    skew = K_INV * pos_ratio

    raw_bid_quote = mid_quote - delta_dynamic / 2 - skew
    raw_ask_quote = mid_quote + delta_dynamic / 2 - skew

    if position > 0:
        raw_bid_quote -= 0.3 * (delta_dynamic / 2)
        raw_ask_quote -= 0.1 * (delta_dynamic / 2)
    elif position < 0:
        raw_bid_quote += 0.1 * (delta_dynamic / 2)
        raw_ask_quote += 0.3 * (delta_dynamic / 2)

    raw_bid_quote = round_to_tick(raw_bid_quote, TICK_SIZE, side="BUY")
    raw_ask_quote = round_to_tick(raw_ask_quote, TICK_SIZE, side="SELL")

    bid_quote = min(raw_bid_quote, best_ask - TICK_SIZE)
    ask_quote = max(raw_ask_quote, best_bid + TICK_SIZE)

    bid_clipped = bid_quote != raw_bid_quote
    ask_clipped = ask_quote != raw_ask_quote

    return pd.Series({
        "spread": spread,
        "delta_dynamic": delta_dynamic,
        "mid_quote": mid_quote,
        "raw_bid_quote": raw_bid_quote,
        "raw_ask_quote": raw_ask_quote,
        "bid_quote": bid_quote,
        "ask_quote": ask_quote,
        "bid_clipped": bid_clipped,
        "ask_clipped": ask_clipped,
    })


def trace_row(row):
    logger.info(
        f"[TRACE] "
        f"{row['team']:12s} | "
        f"Pos={int(row['position']):4d} | "
        f"Book={fmt(row['best_bid'])}/{fmt(row['best_ask'])} | "
        f"Mkt={fmt(row['market_price'])} | "
        f"ModelFV={fmt(row['model_fair_value'])} | "
        f"RawEdge={fmt(row['raw_edge'])} | "
        f"PxChg={fmt(row['price_change'])} | "
        f"Disagree={row['disagreement']} | "
        f"Score={fmt(row['disagree_score'])} | "
        f"Conf={fmt(row['confidence'])} | "
        f"EffFV={fmt(row['effective_fair_value'])} | "
        f"EffEdge={fmt(row['effective_edge'])} | "
        f"Trade={row['trade_candidate']} | "
        f"ExitOnly={row['inventory_exit_candidate']} | "
        f"Mid={fmt(row['mid_quote'])} | "
        f"RawQ={fmt(row['raw_bid_quote'])}/{fmt(row['raw_ask_quote'])} | "
        f"FinalQ={fmt(row['bid_quote'])} x {int(row['bid_size'])} / "
        f"{fmt(row['ask_quote'])} x {int(row['ask_size'])} | "
        f"Clipped={row['bid_clipped']}/{row['ask_clipped']}"
        f"Spr={fmt(row['spread'])} | "
        f"Delta={fmt(row['delta_dynamic'])} | "
    )


def get_open_orders_by_team_and_side(open_orders):
    """
    Map open orders into:
    orders_map[team][side] = {"order_id": ..., "px": ..., "qty": ...}
    """
    orders_map = {}

    for order_id, order in open_orders.items():
        team = order.display_symbol
        side = "BUY" if order.qty > 0 else "SELL"

        if team not in orders_map:
            orders_map[team] = {}

        orders_map[team][side] = {
            "order_id": order_id,
            "px": order.px,
            "qty": abs(order.qty),
        }

    return orders_map


def order_is_stale(current_px, current_qty, desired_px, desired_qty):
    px_changed = abs(current_px - desired_px) >= PRICE_REPLACE_THRESHOLD
    qty_changed = abs(current_qty - desired_qty) > SIZE_REPLACE_THRESHOLD
    return px_changed or qty_changed


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

            # -------- disagreement flag --------
            df["disagreement"] = df.apply(
                lambda row: thesis_disagreement(row["raw_edge"], row["price_change"]),
                axis=1,
            )

            # -------- disagreement score update --------
            for _, row in df.iterrows():
                team = row["team"]
                old_score = disagree_score.get(team, 0.0)

                if row["disagreement"]:
                    new_score = min(old_score + DISAGREE_INCREMENT, MAX_DISAGREE_SCORE)
                else:
                    new_score = max(old_score - DISAGREE_DECAY, 0.0)

                disagree_score[team] = new_score

            df["disagree_score"] = df["team"].map(lambda t: disagree_score.get(t, 0.0))
            df["confidence"] = df["disagree_score"].apply(score_to_confidence)

            # -------- effective fair value / edge --------
            df["effective_fair_value"] = (
                df["market_price"] +
                df["confidence"] * (df["model_fair_value"] - df["market_price"])
            )
            df["effective_edge"] = df["effective_fair_value"] - df["market_price"]
            df["effective_edge_pct"] = df["effective_edge"] / df["market_price"]

            # -------- candidate flags --------
            df["trade_candidate"] = (
                (df["effective_edge"].abs() > EDGE_THRESHOLD) |
                (df["effective_edge_pct"].abs() > EDGE_PCT_THRESHOLD)
            )

            df["inventory_exit_candidate"] = (
                (~df["trade_candidate"]) &
                (df["position"].abs() >= INVENTORY_EXIT_THRESHOLD)
            )

            df = df[df["trade_candidate"] | df["inventory_exit_candidate"]].copy()

            if df.empty:
                logger.info("No trade or inventory-exit candidates this iteration.")

                current_open_orders = await self.get_open_orders()
                if current_open_orders:
                    logger.info("Canceling all open orders (no desired quotes).")
                    try:
                        await self.cancel_orders(list(current_open_orders.keys()))
                    except Exception as e:
                        logger.info(f"Failed to cancel open orders: {e}")

                await asyncio.sleep(SLEEP_SECONDS)
                continue

            # -------- quotes --------
            quote_df = df.apply(build_quotes, axis=1)
            df = pd.concat([df, quote_df], axis=1)

            # -------- sizes --------
            size_df = df.apply(
                lambda row: compute_quote_sizes(
                    edge=row["effective_edge"],
                    position=row["position"],
                ),
                axis=1,
                result_type="expand",
            )
            size_df.columns = ["bid_size", "ask_size"]
            df[["bid_size", "ask_size"]] = size_df

            # -------- inventory-only unwind mode --------
            for idx, row in df.iterrows():
                if row["inventory_exit_candidate"]:
                    pos = int(row["position"])

                    if pos > 0:
                        # long inventory, no alpha left -> only work asks
                        df.at[idx, "bid_size"] = 0
                        df.at[idx, "ask_size"] = min(
                            pos,
                            max(MIN_EXIT_SIZE, int(row["ask_size"]))
                        )

                    elif pos < 0:
                        # short inventory, no alpha left -> only work bids
                        df.at[idx, "ask_size"] = 0
                        df.at[idx, "bid_size"] = min(
                            abs(pos),
                            max(MIN_EXIT_SIZE, int(row["bid_size"]))
                        )

            # -------- trace --------
            if TRACE_ENABLED:
                logger.info("-------- TRACE --------")
                for _, row in df.iterrows():
                    trace_row(row)

            # -------- update price memory --------
            for team, px in zip(df["team"], df["market_price"]):
                prev_prices[team] = px

            # -------- update orders selectively --------
            logger.info("-------- UPDATING ORDERS --------")
            order_type = "LIMIT"

            current_open_orders = await self.get_open_orders()
            orders_map = get_open_orders_by_team_and_side(current_open_orders)

            wanted_pairs = set()

            for _, row in df.iterrows():
                team = row["team"]
                bid_quote = row["bid_quote"]
                ask_quote = row["ask_quote"]
                bid_size = int(row["bid_size"])
                ask_size = int(row["ask_size"])

                # safety checks
                if row["best_bid"] >= row["best_ask"]:
                    logger.info(f"Skipping {team}: invalid book.")
                    continue

                if bid_quote >= ask_quote:
                    logger.info(f"Skipping {team}: crossed internal quote.")
                    continue

                logger.info(
                    f"{team} | ExitOnly={row['inventory_exit_candidate']} | "
                    f"bid_size={bid_size} ask_size={ask_size} | "
                    f"bid_quote={bid_quote:.2f} ask_quote={ask_quote:.2f} | "
                    f"best_bid={row['best_bid']:.2f} best_ask={row['best_ask']:.2f}"
                )

                # BUY side
                if bid_size > 0:
                    wanted_pairs.add((team, "BUY"))
                    current = orders_map.get(team, {}).get("BUY")

                    if current is None:
                        try:
                            await self.send_order(team, bid_quote, bid_size, order_type)
                            logger.info(f"Placed BUY for {team}: {bid_quote:.2f} x {bid_size}")
                        except Exception as e:
                            logger.info(f"Failed to place BUY for {team}: {e}")
                    else:
                        if order_is_stale(current["px"], current["qty"], bid_quote, bid_size):
                            try:
                                await self.cancel_orders([current["order_id"]])
                                await self.send_order(team, bid_quote, bid_size, order_type)
                                logger.info(
                                    f"Replaced BUY for {team}: "
                                    f"{current['px']:.2f} x {current['qty']} -> {bid_quote:.2f} x {bid_size}"
                                )
                            except Exception as e:
                                logger.info(f"Failed to replace BUY for {team}: {e}")
                        else:
                            logger.info(
                                f"Kept BUY for {team}: {current['px']:.2f} x {current['qty']} "
                                f"(desired {bid_quote:.2f} x {bid_size})"
                            )

                # SELL side
                if ask_size > 0:
                    wanted_pairs.add((team, "SELL"))
                    current = orders_map.get(team, {}).get("SELL")

                    if current is None:
                        try:
                            await self.send_order(team, ask_quote, -ask_size, order_type)
                            logger.info(f"Placed SELL for {team}: {ask_quote:.2f} x {ask_size}")
                        except Exception as e:
                            logger.info(f"Failed to place SELL for {team}: {e}")
                    else:
                        if order_is_stale(current["px"], current["qty"], ask_quote, ask_size):
                            try:
                                await self.cancel_orders([current["order_id"]])
                                await self.send_order(team, ask_quote, -ask_size, order_type)
                                logger.info(
                                    f"Replaced SELL for {team}: "
                                    f"{current['px']:.2f} x {current['qty']} -> {ask_quote:.2f} x {ask_size}"
                                )
                            except Exception as e:
                                logger.info(f"Failed to replace SELL for {team}: {e}")
                        else:
                            logger.info(
                                f"Kept SELL for {team}: {current['px']:.2f} x {current['qty']} "
                                f"(desired {ask_quote:.2f} x {ask_size})"
                            )

            # Cancel current orders that are no longer wanted
            for team, side_map in orders_map.items():
                for side, current in side_map.items():
                    if (team, side) not in wanted_pairs:
                        try:
                            await self.cancel_orders([current["order_id"]])
                            logger.info(
                                f"Canceled stale {side} for {team}: "
                                f"{current['px']:.2f} x {current['qty']}"
                            )
                        except Exception as e:
                            logger.info(f"Failed to cancel stale {side} for {team}: {e}")

            print('\n')
            await asyncio.sleep(SLEEP_SECONDS)


async def main():
    async with create_session() as session:
        client = TradingBot(session, game_id, token)
        print(f"Access web view at {client.web_url}")
        await client.start()


if __name__ == "__main__":
    asyncio.run(main())