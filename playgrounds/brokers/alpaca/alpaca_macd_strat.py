import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from alpaca.data.enums import DataFeed
from alpaca.data.live import StockDataStream
from alpaca.data.models.bars import Bar
from dotenv import load_dotenv

from algotrading.lib.alpha.macd import MACDAlpha, MACDAlphaConfig, MACDAlphaOutput
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

V2_SRC = Path(__file__).resolve().parents[3] / "v2" / "src"
if str(V2_SRC) not in sys.path:
    sys.path.insert(0, str(V2_SRC))

from trading_api.alpaca import AlpacaTradingAPI
from models.trading import OrderIntent, OrderSide, OrderType, TimeInForce

TEST_FEED_URL = "wss://stream.data.alpaca.markets/v2/test"
TEST_FEED_SYMBOL = "FAKEPACA"

BASE_TF = Timeframe(1, TimeframeUnit.MINUTE)
ET_TZ = ZoneInfo("America/New_York")
RTH_START = time(9, 30)
RTH_END = time(16, 0)


@dataclass
class PositionState:
    qty: Optional[Decimal] = None
    entry_price: Optional[float] = None
    in_position: bool = False


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


def _normalize_symbol(raw: str) -> str:
    return str(raw).strip().upper().replace(".", "-")


def _sync_position(api: AlpacaTradingAPI, symbol: str, state: PositionState) -> None:
    symbol = _normalize_symbol(symbol)
    positions = api.list_positions()
    pos = next((p for p in positions if p.symbol == symbol), None)
    if pos and pos.qty is not None and pos.qty > 0:
        state.in_position = True
        state.qty = pos.qty
        if pos.avg_entry_price is not None:
            state.entry_price = float(pos.avg_entry_price)
    else:
        state.in_position = False
        state.qty = None
        state.entry_price = None


def _build_intent(
    *,
    symbol: str,
    side: OrderSide,
    qty: Optional[Decimal],
    notional: Optional[Decimal],
) -> OrderIntent:
    return OrderIntent(
        client_order_id=f"macd-{symbol}-{int(datetime.now(timezone.utc).timestamp())}",
        instrument=InstrumentRef(symbol),
        side=side,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.DAY,
        qty=qty,
        notional=notional,
    )


def _to_et(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(ET_TZ)


def _is_rth(ts: datetime) -> bool:
    local = _to_et(ts)
    return RTH_START <= local.time() < RTH_END


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpaca MACD strategy playground")
    parser.add_argument(
        "--test-feed",
        action="store_true",
        help="Use Alpaca test feed (FAKEPACA).",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="",
        help="Single symbol to trade (required unless --test-feed).",
    )
    parser.add_argument(
        "--tf",
        type=str,
        default="1m",
        help="Single aggregation timeframe (e.g. 1m,5m)",
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
    parser.add_argument(
        "--tp-pct",
        type=float,
        default=0.02,
        help="Take-profit percent (e.g. 0.02 for 2%%)",
    )
    parser.add_argument(
        "--sl-pct",
        type=float,
        default=0.01,
        help="Stop-loss percent (e.g. 0.01 for 1%%)",
    )
    parser.add_argument(
        "--time-stop-buffer-mins",
        type=int,
        default=2,
        help="Minutes before close to exit all positions",
    )
    parser.add_argument(
        "--gc-every",
        type=int,
        default=30,
        help="Run aggregator GC every N ticks",
    )
    parser.add_argument(
        "--notional",
        type=float,
        default=0.0,
        help="Buy notional amount (USD).",
    )
    parser.add_argument(
        "--qty",
        type=float,
        default=0.0,
        help="Buy quantity (shares).",
    )
    args = parser.parse_args()

    load_dotenv()

    symbol = _normalize_symbol(args.symbol)
    if args.test_feed:
        symbol = TEST_FEED_SYMBOL
    if not symbol:
        raise SystemExit("--symbol is required when not using the test feed")

    if args.notional <= 0 and args.qty <= 0:
        raise SystemExit("Provide --notional or --qty for buys")
    if args.notional > 0 and args.qty > 0:
        raise SystemExit("Specify only one of --notional or --qty")

    notional = Decimal(str(args.notional)) if args.notional > 0 else None
    qty = Decimal(str(args.qty)) if args.qty > 0 else None

    tp_pct = float(args.tp_pct)
    sl_pct = float(args.sl_pct)
    tf = _parse_timeframe(args.tf)

    api = AlpacaTradingAPI()
    state = PositionState()
    _sync_position(api, symbol, state)

    aggregator = DirectFromBaseAggregator(
        DirectFromBaseAggregatorConfig(source_name="alpaca", base_timeframe=BASE_TF)
    )
    ref = InstrumentRef(symbol)
    aggregator.on_subscribe(
        MDBarSubscribeRequest(
            ts=datetime.now(timezone.utc).timestamp(),
            source="alpaca",
            instrument_refs=(ref,),
            timeframes=(tf,),
        )
    )

    alpha_engine = AlphaEngine()
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
    prev_macd: dict[str, Optional[float]] = {"value": None}
    session_state = {"date": None, "time_stop_triggered": False}
    tick_state = {"count": 0}

    async def bar_data_handler(bar: Bar) -> None:
        # Raw bar logging
        print(
            f"[{datetime.now(timezone.utc).isoformat()}] Received bar: "
            f"{bar.symbol} {bar.timestamp} O:{bar.open} H:{bar.high} L:{bar.low} C:{bar.close} V:{bar.volume}",
            flush=True,
        )

        ohlcv = _bar_to_ohlcv(bar)
        bar_ts = ohlcv.start_ts
        # Session window gating and resets
        if not _is_rth(bar_ts):
            print(
                f"[{datetime.now(timezone.utc).isoformat()}] Bar outside RTH, skipping",
                flush=True,
            )
            return

        session_date = _to_et(bar_ts).date()
        if session_state["date"] != session_date:
            session_state["date"] = session_date
            session_state["time_stop_triggered"] = False
            alpha_engine.reset()
            prev_macd["value"] = None
            print(
                f"[{datetime.now(timezone.utc).isoformat()}] "
                f"New session detected ({session_date}); MACD reset",
                flush=True,
            )

        stop_time = (
            datetime.combine(session_date, RTH_END, ET_TZ)
            - timedelta(minutes=int(args.time_stop_buffer_mins))
        ).time()
        if (
            not session_state["time_stop_triggered"]
            and _to_et(bar_ts).time() >= stop_time
        ):
            # Time-stop exit
            _sync_position(api, symbol, state)
            if state.in_position and state.qty:
                intent = _build_intent(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    qty=state.qty,
                    notional=None,
                )
                api.submit_order(intent)
                print(
                    f"[{datetime.now(timezone.utc).isoformat()}] "
                    f"Time stop exit, selling {symbol}",
                    flush=True,
                )
                _sync_position(api, symbol, state)
            session_state["time_stop_triggered"] = True
            return
        # Build base event + derive aggregated bars
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
            if not isinstance(ev, BarCompleted):
                continue

            # Bar completion -> alpha update
            print(
                f"[{datetime.now(timezone.utc).isoformat()}] Derived event: {ev}",
                flush=True,
            )

            outputs = alpha_engine.update(ev)
            output: Optional[MACDAlphaOutput] = next(iter(outputs.values()), None)
            if output is None or not output.is_ready:
                print(
                    f"[{datetime.now(timezone.utc).isoformat()}] MACD output not ready",
                    flush=True,
                )
                return

            macd_value = float(output.macd)
            print(
                f"[{datetime.now(timezone.utc).isoformat()}] MACD value: {macd_value:.5f}",
                flush=True,
            )
            if prev_macd["value"] is None:
                prev_macd["value"] = macd_value
                return

            # Position sync + risk checks
            _sync_position(api, symbol, state)
            price = float(ev.bar.c)

            if state.in_position and state.entry_price:
                pnl = (price - state.entry_price) / state.entry_price
                qty_float = float(state.qty) if state.qty is not None else 0.0
                unrealized = (price - state.entry_price) * qty_float
                print(
                    f"[{datetime.now(timezone.utc).isoformat()}] "
                    f"Position qty={state.qty} pnl={pnl:.4f} "
                    f"unrlzd=${unrealized:.2f} price={price:.2f} "
                    f"entry={state.entry_price:.2f}",
                    flush=True,
                )
                if pnl >= tp_pct:
                    intent = _build_intent(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        qty=state.qty,
                        notional=None,
                    )
                    api.submit_order(intent)
                    print(
                        f"[{datetime.now(timezone.utc).isoformat()}] "
                        f"TP hit, selling {symbol} at {price:.2f}",
                        flush=True,
                    )
                    _sync_position(api, symbol, state)
                elif pnl <= -sl_pct:
                    intent = _build_intent(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        qty=state.qty,
                        notional=None,
                    )
                    api.submit_order(intent)
                    print(
                        f"[{datetime.now(timezone.utc).isoformat()}] "
                        f"SL hit, selling {symbol} at {price:.2f}",
                        flush=True,
                    )
                    _sync_position(api, symbol, state)

            # Signal-based entries/exits
            if not state.in_position and prev_macd["value"] <= 0 and macd_value > 0:
                intent = _build_intent(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    qty=qty,
                    notional=notional,
                )
                api.submit_order(intent)
                state.entry_price = price
                print(
                    f"[{datetime.now(timezone.utc).isoformat()}] "
                    f"MACD entry, buying {symbol} at {price:.2f}",
                    flush=True,
                )
                _sync_position(api, symbol, state)
            elif state.in_position and prev_macd["value"] > 0 and macd_value <= 0:
                intent = _build_intent(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    qty=state.qty,
                    notional=None,
                )
                api.submit_order(intent)
                print(
                    f"[{datetime.now(timezone.utc).isoformat()}] "
                    f"MACD exit, selling {symbol} at {price:.2f}",
                    flush=True,
                )
                _sync_position(api, symbol, state)

            prev_macd["value"] = macd_value

        # Aggregator GC
        tick_state["count"] += 1
        if args.gc_every > 0 and tick_state["count"] % args.gc_every == 0:
            print(
                f"[{datetime.now(timezone.utc).isoformat()}] Running aggregator GC",
                flush=True,
            )
            gc_events = aggregator.run_gc(now=datetime.now(timezone.utc))
            gc_closed = [ev for ev in gc_events if isinstance(ev, BarClosed)]
            if gc_closed:
                print(
                    f"[{datetime.now(timezone.utc).isoformat()}] "
                    f"GC closed {len(gc_closed)} bars",
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
        wss_client.subscribe_bars(bar_data_handler, symbol)

    wss_client.run()
