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


# global params

ALPHA = 0.80              # weight toward market
DELTA = 2.0               # total spread width
K_INV = 2.5               # inventory skew strength
# K_INV = 1.25              # inventory skew strength
MAX_POSITION = 60

EDGE_THRESHOLD = 1.0
EDGE_PCT_THRESHOLD = 0.10

BASE_SIZE = 20
EDGE_SCALE = 10.0

TICK_SIZE = 0.01
SLEEP_SECONDS = 30


# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(handler)
# disable all non-critical (omit for now, uncomment for execution)
# logging.disable(logging.CRITICAL)


# helpers

def round_to_tick(px: float, tick_size: float = TICK_SIZE, side: str | None = None) -> float:
    """
    Round a price to the exchange tick.

    side:
        - 'BUY'  -> round down so you do not bid more than intended
        - 'SELL' -> round up so you do not offer lower than intended
        - None   -> round to nearest tick
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
    Compute mid prices only for teams where we have fair values.
    """
    asks, bids, out = {}, {}, {}
    for team in fair_values:
        book = order_books[team]
        asks[team] = book.best_ask_px
        bids[team] = book.best_bid_px
        out[team] = (book.best_bid_px + book.best_ask_px) / 2
    return asks, bids, out


def compute_quote_sizes(edge: float, position: int) -> tuple[int, int]:
    """
    Returns (bid_size, ask_size).

    Logic:
    - scale with edge magnitude
    - if move is adverse, reduce aggression
    - if already long, reduce bids and preserve/encourage asks
    - if already short, reduce asks and preserve/encourage bids
    - enforce hard inventory caps
    """
    edge_strength = min(abs(edge) / EDGE_SCALE, 1.0)
    base = BASE_SIZE * edge_strength

    pos_ratio = min(abs(position) / MAX_POSITION, 1.0)

    bid_size = 0.0
    ask_size = 0.0

    if edge > 0:
        # Model likes owning more, but inventory still matters.
        # As long inventory grows, reduce bids sharply.
        bid_size = base * (1.0 - pos_ratio)
        ask_size = base * (0.5 + 0.5 * pos_ratio)

    elif edge < 0:
        # Model likes being shorter, but inventory still matters.
        # As short inventory grows, reduce asks sharply.
        bid_size = base * (0.5 + 0.5 * pos_ratio)
        ask_size = base * (1.0 - pos_ratio)

    # Hard caps
    if position >= MAX_POSITION:
        bid_size = 0.0
    if position <= -MAX_POSITION:
        ask_size = 0.0

    return int(bid_size), int(ask_size)


def build_quotes(row):
    """
    Build inventory-aware bid/ask quotes for one symbol.
    """
    fair_value = row["fair_value"]
    market_price = row["market_price"]
    position = row["position"]
    best_ask = row["best_ask"]
    best_bid = row["best_bid"]

    # One unified quote center formula
    mid_quote = fair_value + ALPHA * (market_price - fair_value)

    # Inventory skew
    pos_ratio = position / MAX_POSITION
    skew = K_INV * pos_ratio

    bid_quote = mid_quote - DELTA / 2 - skew
    ask_quote = mid_quote + DELTA / 2 - skew

    # asymmetric quoting (ie not just even spacing on either side of mid)

    # mellow enforcement, should allow
    # you to not have to dump inventory at
    # worse price than entrance
    if position > 0:
        bid_quote -= 0.3 * (DELTA / 2)
        ask_quote -= 0.1 * (DELTA / 2)
    elif position < 0:
        bid_quote += 0.1 * (DELTA / 2)
        ask_quote += 0.3 * (DELTA / 2)

    # # use below when accumulating too much inventory
    # # but comment out when closing positions
    # # at worse price than entering into 
    # if position > 0:
    #     bid_quote -= 0.5 * (DELTA / 2)
    #     ask_quote -= 0.2 * (DELTA / 2)
    # elif position < 0:
    #     bid_quote += 0.2 * (DELTA / 2)
    #     ask_quote += 0.5 * (DELTA / 2)

    bid_quote = round_to_tick(bid_quote, TICK_SIZE, side="BUY")
    ask_quote = round_to_tick(ask_quote, TICK_SIZE, side="SELL")

    # always ensure ask is greater than highest bid
    ask_quote = max(ask_quote, best_bid + TICK_SIZE)

    # always ensure bid is less than lowest ask
    bid_quote = min(bid_quote, best_ask - TICK_SIZE)


    return pd.Series({
        "mid_quote": mid_quote,
        "bid_quote": bid_quote,
        "ask_quote": ask_quote,
    })


# client 

class TradingBot(Client):
    def __init__(
        self,
        session: aiohttp.ClientSession,
        game_id: int,
        token: str,
    ) -> None:
        super().__init__(session, game_id, token)

    async def on_start(self) -> None:

        while True:
            order_books = await self.get_order_books()
            positions = self.positions

            # -------- fair values and market prices --------
            fair_value_df = compute_ev_df()
            fair_values = dict(zip(fair_value_df["team"], fair_value_df["EV"]))
            asks, bids, market_prices = mid_price_map(order_books, fair_values)

            common_teams = sorted(set(fair_values) & set(market_prices))

            df = pd.DataFrame({"team": common_teams})
            df["fair_value"] = df["team"].map(fair_values)
            df["best_ask"] = df["team"].map(asks)
            df["best_bid"] = df["team"].map(bids)
            df["market_price"] = df["team"].map(market_prices)
            df["position"] = df["team"].map(lambda t: positions.get(t, 0))

            # -------- edge --------
            df["edge"] = df["fair_value"] - df["market_price"]
            df["edge_pct"] = df["edge"] / df["market_price"]

            df["trade_candidate"] = (
                (df["edge"].abs() > EDGE_THRESHOLD) |
                (df["edge_pct"].abs() > EDGE_PCT_THRESHOLD)
            )

            df = df[df["trade_candidate"]].copy()

            # -------- quotes --------
            quote_df = df.apply(build_quotes, axis=1)

            df = pd.concat([df, quote_df], axis=1)

            # -------- sizes --------
            df[["bid_size", "ask_size"]] = df.apply(
                lambda row: pd.Series(
                    compute_quote_sizes(
                        edge=row["edge"],
                        position=row["position"],
                    )
                ),
                axis=1,
            )

            # cancel all open orders every iteration
            logger.info("-------- CANCELING ALL OPEN ORDERS --------")
            current_open_orders = await self.get_open_orders()
            if current_open_orders:
                try:
                    await self.cancel_orders(list(current_open_orders.keys()))
                except Exception as e:
                    logger.info(f"Failed to cancel open orders: {e}")

            # place fresh quotes 
            logger.info("-------- SENDING NEW ORDERS --------")
            order_type = "LIMIT"

            for _, row in df.iterrows():
                team = row["team"]
                market_price = row["market_price"]
                edge = row["edge"]

                bid_quote = row["bid_quote"]
                ask_quote = row["ask_quote"]
                bid_size = int(row["bid_size"])
                ask_size = int(row["ask_size"])

                if bid_size > 0:
                    try:
                        await self.send_order(team, bid_quote, bid_size, order_type)
                        logger.info(
                            f"BUY  | Team: {team:12s} | Qty: {bid_size:3d} | "
                            f"Market: {market_price:.2f} | Px: {bid_quote:.2f} | Edge: {edge:.2f}"
                        )
                    except Exception as e:
                        logger.info(f"Failed to place BUY for {team}: {e}")

                if ask_size > 0:
                    try:
                        await self.send_order(team, ask_quote, -ask_size, order_type)
                        logger.info(
                            f"SELL | Team: {team:12s} | Qty: {ask_size:3d} | "
                            f"Market: {market_price:.2f} | Px: {ask_quote:.2f} | Edge: {edge:.2f}"
                        )
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
