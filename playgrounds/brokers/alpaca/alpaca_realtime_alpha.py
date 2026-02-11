import argparse
import os
from datetime import datetime, timedelta, timezone
from typing import List

from alpaca.data.enums import DataFeed
from alpaca.data.live import StockDataStream
from alpaca.data.models.bars import Bar
from dotenv import load_dotenv

from algotrading.lib.alpha.ema import EMAAlpha, EMAAlphaConfig
from algotrading.lib.alpha.macd import MACDAlpha, MACDAlphaConfig
from algotrading.lib.alpha_engine.engine import AlphaEngine
from algotrading.lib.eventing.md_events import (
    BaseBarUpserted,
    BarClosed,
    BarCompleted,
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


def _parse_timeframe(raw: str) -> Timeframe:
    item = raw.strip()
    if len(item) < 2:
        raise ValueError(f"Invalid timeframe: {item}")
    unit_map = {
        "s": TimeframeUnit.SECOND,
        "m": TimeframeUnit.MINUTE,
        "h": TimeframeUnit.HOUR,
        "d": TimeframeUnit.DAY,
    }
    n_part, unit_part = item[:-1], item[-1].lower()
    if unit_part not in unit_map:
        raise ValueError(f"Invalid timeframe unit: {unit_part}")
    try:
        n = int(n_part)
    except ValueError as exc:
        raise ValueError(f"Invalid timeframe value: {n_part}") from exc
    if n <= 0:
        raise ValueError(f"Invalid timeframe value: {n}")
    return Timeframe(n, unit_map[unit_part])


def _ensure_tzaware(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _bar_to_ohlcv(bar: Bar) -> OHLCVBar:
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


def _parse_symbols(raw: str, *, test_feed: bool) -> List[str]:
    items = [s.strip().upper() for s in raw.split(",") if s.strip()]
    if test_feed:
        if not items:
            return [TEST_FEED_SYMBOL]
        if any(sym != TEST_FEED_SYMBOL for sym in items):
            raise SystemExit(
                f"Test feed only supports {TEST_FEED_SYMBOL}; remove other symbols."
            )
        return items

    if not items:
        raise SystemExit("--symbols is required when not using the test feed")
    return items


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpaca realtime alpha playground")
    parser.add_argument(
        "--test-feed",
        action="store_true",
        help="Use Alpaca test feed (FAKEPACA).",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="",
        help=f"Comma-separated symbols (test feed supports only {TEST_FEED_SYMBOL})",
    )
    parser.add_argument(
        "--tf",
        type=str,
        default="1m",
        help="Single aggregation timeframe (e.g. 1m,5m)",
    )
    parser.add_argument(
        "--ema-window",
        type=int,
        default=20,
        help="EMA window length",
    )
    parser.add_argument(
        "--alpha",
        type=str,
        default="ema",
        choices=("ema", "macd"),
        help="Alpha type to compute",
    )
    parser.add_argument(
        "--macd-ma-type",
        type=str,
        default="ema",
        choices=("ema", "sma"),
        help="MACD underlying MA type",
    )
    parser.add_argument(
        "--macd-fast-window",
        type=int,
        default=12,
        help="MACD fast window",
    )
    parser.add_argument(
        "--macd-slow-window",
        type=int,
        default=26,
        help="MACD slow window",
    )
    parser.add_argument(
        "--macd-signal-window",
        type=int,
        default=9,
        help="MACD signal window",
    )
    args = parser.parse_args()

    load_dotenv()

    symbols = _parse_symbols(args.symbols, test_feed=args.test_feed)
    tf = _parse_timeframe(args.tf)

    instrument_refs = tuple(InstrumentRef(sym) for sym in symbols)
    aggregator = DirectFromBaseAggregator(
        DirectFromBaseAggregatorConfig(source_name="alpaca", base_timeframe=BASE_TF)
    )
    aggregator.on_subscribe(
        MDBarSubscribeRequest(
            ts=datetime.now(timezone.utc).timestamp(),
            source="alpaca",
            instrument_refs=instrument_refs,
            timeframes=(tf,),
        )
    )

    alpha_engine = AlphaEngine()
    for ref in instrument_refs:
        if args.alpha == "ema":
            alpha_engine.subscribe(
                ref=ref,
                tf=tf,
                alpha_type=EMAAlpha,
                config=EMAAlphaConfig(ref=ref, tf=tf, window=args.ema_window),
            )
        else:
            alpha_engine.subscribe(
                ref=ref,
                tf=tf,
                alpha_type=MACDAlpha,
                config=MACDAlphaConfig(
                    ref=ref,
                    tf=tf,
                    ma_type=args.macd_ma_type,
                    fast_window=args.macd_fast_window,
                    slow_window=args.macd_slow_window,
                    signal_window=args.macd_signal_window,
                ),
            )

    async def bar_data_handler(bar: Bar) -> None:
        ohlcv = _bar_to_ohlcv(bar)
        ref = InstrumentRef(bar.symbol)
        key = BarKey(ref=ref, tf=BASE_TF, start_ts=ohlcv.start_ts)
        base_ev = BaseBarUpserted(
            ts=datetime.now(timezone.utc).timestamp(),
            source="alpaca",
            key=key,
            curr=ohlcv,
            prev=None,
            is_correction=False,
        )

        derived_events = aggregator.on_base_upsert(base_ev)
        for ev in derived_events:
            # Update only on bar completion for now, ignoring bar updates / corrections / closures
            if not isinstance(ev, (BarCompleted)):
                continue

            # Print completed bar event
            print(
                f"[{datetime.now(timezone.utc).isoformat()}] Derived event: {ev}",
                flush=True,
            )

            # Feed completed bar events into the alpha engine and print outputs
            outputs = alpha_engine.update(ev)
            for alpha_key, output in outputs.items():
                if not output.is_ready:
                    print(
                        f"[{datetime.now(timezone.utc).isoformat()}] "
                        f"{alpha_key.ref.symbol} {alpha_key.tf} "
                        f"{alpha_key.alpha_type.__name__}=not ready",
                        flush=True,
                    )
                    continue
                if alpha_key.alpha_type is EMAAlpha:
                    print(
                        f"[{datetime.now(timezone.utc).isoformat()}] "
                        f"{alpha_key.ref.symbol} {alpha_key.tf} "
                        f"EMA={output.value:.4f}",
                        flush=True,
                    )
                else:
                    print(
                        f"[{datetime.now(timezone.utc).isoformat()}] "
                        f"{alpha_key.ref.symbol} {alpha_key.tf} "
                        f"MACD={output.macd:.4f} signal={output.signal:.4f}",
                        flush=True,
                    )

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
