import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, List

from alpaca.data.enums import DataFeed
from alpaca.data.live import StockDataStream
from dotenv import load_dotenv

from algotrading.lib.eventing.md_events import BaseBarUpserted
from algotrading.lib.market_data.realtime.aggregator.direct_from_base import (
    DirectFromBaseAggregator,
    DirectFromBaseAggregatorConfig,
)
from algotrading.lib.types.instruments import InstrumentRef
from algotrading.lib.types.market_data import BarKey, OHLCVBar, Timeframe, TimeframeUnit

TEST_FEED_URL = "wss://stream.data.alpaca.markets/v2/test"
TEST_FEED_SYMBOL = "FAKEPACA"

BASE_TF = Timeframe(1, TimeframeUnit.MINUTE)


@dataclass(frozen=True)
class _SubMsg:
    refs: tuple[InstrumentRef, ...]
    timeframes: tuple[Timeframe, ...]


def _parse_timeframes(raw: str) -> List[Timeframe]:
    items = [s.strip() for s in raw.split(",") if s.strip()]
    if not items:
        raise ValueError("At least one timeframe is required")
    tfs: List[Timeframe] = []
    unit_map = {
        "s": TimeframeUnit.SECOND,
        "m": TimeframeUnit.MINUTE,
        "h": TimeframeUnit.HOUR,
        "d": TimeframeUnit.DAY,
    }
    for item in items:
        if len(item) < 2:
            raise ValueError(f"Invalid timeframe: {item}")
        n_part, unit_part = item[:-1], item[-1].lower()
        if unit_part not in unit_map:
            raise ValueError(f"Invalid timeframe unit: {unit_part}")
        try:
            n = int(n_part)
        except ValueError as exc:
            raise ValueError(f"Invalid timeframe value: {n_part}") from exc
        if n <= 0:
            raise ValueError(f"Invalid timeframe value: {n}")
        tfs.append(Timeframe(n, unit_map[unit_part]))
    return tfs


def _ensure_tzaware(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _bar_to_ohlcv(bar) -> OHLCVBar:
    start_ts = _ensure_tzaware(bar.timestamp)
    end_ts = start_ts + timedelta(minutes=1)
    return OHLCVBar(
        start_ts=start_ts,
        end_ts=end_ts,
        o=float(bar.open),
        h=float(bar.high),
        l=float(bar.low),
        c=float(bar.close),
        v=float(bar.volume),
    )


def _print_events(events: Iterable[object]) -> None:
    for ev in events:
        key = getattr(ev, "key", None)
        bar = getattr(ev, "bar", None)
        print(f"{type(ev).__name__}: key={key} bar={bar}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Alpaca realtime market data aggregation playground"
    )
    parser.add_argument(
        "--test-feed",
        action="store_true",
        help="Use Alpaca test feed (FAKEPACA). If set, IEX symbols are ignored.",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="",
        help="Comma-separated symbols to subscribe (IEX only).",
    )
    parser.add_argument(
        "--agg-tfs",
        type=str,
        default="5m",
        help="Comma-separated aggregation timeframes (e.g. 5m,15m).",
    )
    args = parser.parse_args()

    # load env variables from .env file
    load_dotenv()

    agg_tfs = _parse_timeframes(args.agg_tfs)

    if args.test_feed:
        if args.symbols:
            raise SystemExit("--symbols is only applicable when using the IEX feed")
        symbols = [TEST_FEED_SYMBOL]
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if not symbols:
            raise SystemExit("--symbols is required when using the IEX feed")

    refs = tuple(InstrumentRef(sym) for sym in symbols)
    cfg = DirectFromBaseAggregatorConfig(source_name="alpaca", base_timeframe=BASE_TF)
    aggregator = DirectFromBaseAggregator(cfg)
    aggregator.on_subscribe(_SubMsg(refs=refs, timeframes=tuple(agg_tfs)))

    async def bar_data_handler(bar):
        print(
            f"[{datetime.now(timezone.utc).isoformat()}] Received raw bar: {bar}",
            flush=True,
        )
        ohlcv = _bar_to_ohlcv(bar)
        ref = InstrumentRef(bar.symbol)
        key = BarKey(ref=ref, tf=BASE_TF, start_ts=ohlcv.start_ts)
        ev = BaseBarUpserted(
            ts=datetime.now(timezone.utc).timestamp(),
            source="alpaca",
            key=key,
            curr=ohlcv,
            prev=None,
            is_correction=False,
        )
        outs = aggregator.on_base_upsert(ev)
        print(
            f"[{datetime.now(timezone.utc).isoformat()}] Derived events:",
            flush=True,
        )
        _print_events(outs)

    if args.test_feed:
        wss_client = StockDataStream(
            api_key=os.getenv("ALPACA_API_KEY"),
            secret_key=os.getenv("ALPACA_SECRET_KEY"),
            url_override=TEST_FEED_URL,
        )
        wss_client.subscribe_bars(bar_data_handler, TEST_FEED_SYMBOL)
    else:
        wss_client = StockDataStream(
            api_key=os.getenv("ALPACA_API_KEY"),
            secret_key=os.getenv("ALPACA_SECRET_KEY"),
            feed=DataFeed.IEX,
        )
        for sym in symbols:
            wss_client.subscribe_bars(bar_data_handler, sym)

    wss_client.run()
