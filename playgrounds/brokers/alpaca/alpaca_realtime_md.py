import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, List

from alpaca.data.enums import DataFeed
from alpaca.data.live import StockDataStream
from dotenv import load_dotenv

from algotrading.lib.eventing.md_events import (
    BaseBarUpserted,
    BarClosed,
    MDBarBatchSubscribeRequest,
    MDBarSubscribeRequest,
)
from algotrading.lib.market_data.realtime.aggregator.direct_from_base import (
    DirectFromBaseAggregator,
    DirectFromBaseAggregatorConfig,
)
from algotrading.lib.types.instruments import InstrumentRef
from algotrading.lib.types.market_data import BarKey, OHLCVBar, Timeframe, TimeframeUnit

TEST_FEED_URL = "wss://stream.data.alpaca.markets/v2/test"
TEST_FEED_SYMBOL = "FAKEPACA"

BASE_TF = Timeframe(1, TimeframeUnit.MINUTE)


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


def _parse_single_timeframe(raw: str) -> Timeframe:
    tfs = _parse_timeframes(raw)
    if len(tfs) != 1:
        raise ValueError("Exactly one timeframe is required for batch subscription")
    return tfs[0]


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
    parser.add_argument(
        "--batch-symbols",
        type=str,
        default="",
        help="Comma-separated symbols for a single batch group (e.g. AAPL,MSFT).",
    )
    parser.add_argument(
        "--batch-tf",
        type=str,
        default="5m",
        help="Single batch timeframe (e.g. 5m).",
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
        has_batch_symbols = any(
            s.strip() for s in args.batch_symbols.split(",") if s.strip()
        )
        if not symbols and not has_batch_symbols:
            raise SystemExit("--symbols is required when using the IEX feed")

    batch_symbols_raw = [
        s.strip().upper() for s in args.batch_symbols.split(",") if s.strip()
    ]
    instrument_refs = tuple(InstrumentRef(sym) for sym in symbols)
    cfg = DirectFromBaseAggregatorConfig(source_name="alpaca", base_timeframe=BASE_TF)
    aggregator = DirectFromBaseAggregator(cfg)
    aggregator.on_subscribe(
        MDBarSubscribeRequest(
            ts=datetime.now(timezone.utc).timestamp(),
            source="alpaca",
            instrument_refs=instrument_refs,
            timeframes=tuple(agg_tfs),
        )
    )

    if args.batch_symbols or args.batch_tf:
        if not args.batch_tf:
            raise SystemExit("--batch-tf is required when using batch subscription")

        batch_tf = _parse_single_timeframe(args.batch_tf)
        if args.test_feed:
            batch_symbols = [TEST_FEED_SYMBOL]
        else:
            batch_symbols = batch_symbols_raw
            if not batch_symbols:
                raise SystemExit(
                    "--batch-symbols is required when using batch subscription"
                )

        batch_refs = tuple(InstrumentRef(sym) for sym in batch_symbols)
        aggregator.on_subscribe_batch(
            MDBarBatchSubscribeRequest(
                ts=datetime.now(timezone.utc).timestamp(),
                source="alpaca",
                instrument_refs=batch_refs,
                timeframe=batch_tf,
                auto_subscribe_constituents=True,
            )
        )

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

        gc_events = aggregator.run_gc(now=datetime.now(timezone.utc))
        gc_closed = [ev for ev in gc_events if isinstance(ev, BarClosed)]
        if gc_closed:
            print(
                f"[{datetime.now(timezone.utc).isoformat()}] GC closed events:",
                flush=True,
            )
            _print_events(gc_closed)

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
        subscribe_symbols = set(symbols)
        subscribe_symbols.update(batch_symbols_raw)
        for sym in sorted(subscribe_symbols):
            wss_client.subscribe_bars(bar_data_handler, sym)

    wss_client.run()
