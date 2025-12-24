"""Alpaca simple strategy (end-to-end test).

Implements a small polling-based momentum strategy:
- Trades only during market open hours (guarded by the trading clock)
- Trades a very small universe (default: QQQ, SPY)
- Observes price by polling latest trade
- Enters long (fixed notional) when last X observed prices trend up
- Exits on take-profit / stop-loss / max holding period
- Exits all open positions near market close (buffer)
- Runs at a fixed interval (default: 30s)

Credentials are read from environment variables:
- ALPACA_API_KEY
- ALPACA_SECRET_KEY
- ALPACA_BASE_URL (optional; defaults to paper endpoint)
- ALPACA_DATA_FEED (optional: iex|sip; defaults to iex)
- ALPACA_PAPER (optional: true|false; defaults to true)

This is intentionally minimal and designed as an end-to-end test in paper trading.
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone


def _configure_unbuffered_output() -> None:
    """Best-effort: make stdout/stderr stream in real time when redirected."""

    for stream in (sys.stdout, sys.stderr):
        if stream is None:
            continue
        try:
            # Python 3.7+: ensure newline flush + flush on every write.
            stream.reconfigure(line_buffering=True, write_through=True)
        except Exception:
            # If reconfigure isn't supported, we rely on the environment
            # (e.g., `python -u` / `PYTHONUNBUFFERED=1`) and newline prints.
            pass


try:
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestTradeRequest
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
except Exception:
    print("Missing dependency: install with `pip install alpaca-py`", file=sys.stderr)
    raise


@dataclass(frozen=True)
class StrategyConfig:
    universe: list[str] = field(default_factory=lambda: ["QQQ", "SPY", "NVDA", "MU"])
    poll_interval_seconds: int = 30
    lookback_prices: int = 4
    trade_notional_usd: float = 100.0
    take_profit_pct: float = 0.006
    stop_loss_pct: float = 0.005
    max_holding_minutes: int = 60
    close_buffer_minutes: int = 10
    data_feed: str = "iex"  # iex|sip
    max_cycles: int | None = None  # set to an int to stop after N loops


@dataclass
class PositionState:
    entry_price: float
    entry_time: datetime


def _read_env(cfg: StrategyConfig):
    return {
        "api_key": os.environ.get("ALPACA_API_KEY"),
        "secret_key": os.environ.get("ALPACA_SECRET_KEY"),
        "base_url": os.environ.get(
            "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
        ),
        "paper": os.environ.get("ALPACA_PAPER", "true").strip().lower()
        in {"1", "true", "yes"},
        "data_feed": os.environ.get("ALPACA_DATA_FEED", cfg.data_feed),
    }


def _parse_data_feed(feed_str: str | None):
    if not feed_str:
        return None
    s = feed_str.strip().lower()
    if s == "iex":
        return DataFeed.IEX
    if s == "sip":
        return DataFeed.SIP
    return None


def _ensure_dt_utc(dt: datetime | None) -> datetime:
    if dt is None:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        # Alpaca docs: timezone-naive is assumed UTC.
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _trend_up(prices: deque[float]) -> bool:
    if len(prices) < 2:
        return False
    it = iter(prices)
    prev = next(it)
    for cur in it:
        if cur <= prev:
            return False
        prev = cur
    return True


def _get_open_positions_by_symbol(trading: TradingClient) -> dict[str, dict]:
    out: dict[str, dict] = {}
    try:
        positions = trading.get_all_positions()
    except Exception:
        positions = []
    for p in positions:
        raw = getattr(p, "_raw", None)
        if raw is None:
            raw = {
                "symbol": getattr(p, "symbol", None),
                "qty": getattr(p, "qty", None),
                "avg_entry_price": getattr(p, "avg_entry_price", None),
                "side": getattr(p, "side", None),
                "market_value": getattr(p, "market_value", None),
            }
        sym = raw.get("symbol")
        if sym:
            out[str(sym).upper()] = raw
    return out


def _verify_universe_latest_trades(
    data: StockHistoricalDataClient,
    universe: list[str],
    feed: DataFeed | None,
) -> dict[str, float]:
    """Fail fast if any ticker can't return a latest trade.

    This runs once at startup so typos / unsupported tickers are caught even
    if the market is closed.
    """

    try:
        trades = data.get_stock_latest_trade(
            StockLatestTradeRequest(symbol_or_symbols=universe, feed=feed)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to fetch latest trades for universe: {e}") from e

    missing: list[str] = []
    prices: dict[str, float] = {}
    for symbol in universe:
        t = trades.get(symbol)
        price = getattr(t, "price", None)
        if price is None:
            missing.append(symbol)
            continue
        prices[symbol] = float(price)

    if missing:
        raise RuntimeError(
            "Universe validation failed (no latest trade returned) for: "
            + ", ".join(missing)
        )

    return prices


def _submit_market_buy_notional(
    trading: TradingClient, symbol: str, notional_usd: float
):
    req = MarketOrderRequest(
        symbol=symbol,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        notional=notional_usd,
    )
    return trading.submit_order(req)


def _close_position(trading: TradingClient, symbol: str):
    # close_position raises if no position exists.
    return trading.close_position(symbol)


def run_strategy(cfg: StrategyConfig):
    creds = _read_env(cfg)
    if not creds["api_key"] or not creds["secret_key"]:
        raise RuntimeError(
            "Environment variables ALPACA_API_KEY and ALPACA_SECRET_KEY must be set"
        )

    print(f"Trading mode: {'PAPER' if creds['paper'] else 'LIVE'}")

    trading = TradingClient(
        api_key=creds["api_key"],
        secret_key=creds["secret_key"],
        paper=creds["paper"],
        url_override=creds["base_url"],
    )
    data = StockHistoricalDataClient(creds["api_key"], creds["secret_key"])
    feed = _parse_data_feed(creds.get("data_feed"))

    universe = [s.strip().upper() for s in cfg.universe if s and s.strip()]
    if not universe:
        raise RuntimeError("Universe is empty")

    # Validate tickers upfront (works even if market is closed).
    # This catches typos / unsupported symbols early.
    startup_prices = _verify_universe_latest_trades(data, universe, feed)
    print(
        "Startup universe check OK: "
        + ", ".join([f"{s}={startup_prices[s]:.2f}" for s in universe])
    )

    price_windows: dict[str, deque[float]] = {
        s: deque(maxlen=cfg.lookback_prices) for s in universe
    }
    state: dict[str, PositionState] = {}

    cycle = 0
    while True:
        cycle += 1
        if cfg.max_cycles is not None and cycle > cfg.max_cycles:
            print("Reached max_cycles; stopping.")
            return

        clock = trading.get_clock()
        is_open = bool(getattr(clock, "is_open", False))
        now = _ensure_dt_utc(getattr(clock, "timestamp", None))
        next_open = _ensure_dt_utc(getattr(clock, "next_open", None))
        next_close = _ensure_dt_utc(getattr(clock, "next_close", None))

        if not is_open:
            secs = max(30.0, min(300.0, (next_open - now).total_seconds()))
            print(f"[{now.isoformat()}] Market closed; sleeping {int(secs)}s")
            time.sleep(secs)
            continue

        seconds_to_close = (next_close - now).total_seconds()
        close_buffer_seconds = cfg.close_buffer_minutes * 60
        if seconds_to_close <= close_buffer_seconds:
            # Exit all open positions (in our universe) close to market close.
            open_positions = _get_open_positions_by_symbol(trading)
            for symbol in universe:
                if symbol in open_positions:
                    try:
                        _close_position(trading, symbol)
                        print(f"[{now.isoformat()}] Close buffer: closing {symbol}")
                    except Exception as e:
                        print(
                            f"[{now.isoformat()}] Close buffer: failed to close {symbol}: {e}"
                        )
                    state.pop(symbol, None)

            time.sleep(cfg.poll_interval_seconds)
            continue

        # 1) Poll latest trades for universe
        try:
            trades = data.get_stock_latest_trade(
                StockLatestTradeRequest(symbol_or_symbols=universe, feed=feed)
            )
        except Exception as e:
            print(f"[{now.isoformat()}] Latest trade poll failed: {e}")
            time.sleep(cfg.poll_interval_seconds)
            continue

        latest_prices: dict[str, float] = {}
        for symbol in universe:
            t = trades.get(symbol)
            price = getattr(t, "price", None)
            if price is None:
                continue
            latest_prices[symbol] = float(price)
            price_windows[symbol].append(float(price))

        # 2) Determine current positions from broker
        open_positions = _get_open_positions_by_symbol(trading)

        # 3) Exit logic (TP/SL/max hold)
        for symbol in universe:
            if symbol not in open_positions:
                state.pop(symbol, None)
                continue

            current_price = latest_prices.get(symbol)
            if current_price is None:
                continue

            # Ensure we have entry state.
            st = state.get(symbol)
            if st is None:
                raw = open_positions[symbol]
                avg_entry = raw.get("avg_entry_price")
                try:
                    entry_price = (
                        float(avg_entry) if avg_entry is not None else current_price
                    )
                except Exception:
                    entry_price = current_price
                st = PositionState(entry_price=entry_price, entry_time=now)
                state[symbol] = st

            tp_price = st.entry_price * (1.0 + cfg.take_profit_pct)
            sl_price = st.entry_price * (1.0 - cfg.stop_loss_pct)
            held_for = now - st.entry_time
            if current_price >= tp_price:
                reason = "take_profit"
            elif current_price <= sl_price:
                reason = "stop_loss"
            elif held_for >= timedelta(minutes=cfg.max_holding_minutes):
                reason = "max_hold"
            else:
                reason = ""

            if reason:
                try:
                    _close_position(trading, symbol)
                    print(
                        f"[{now.isoformat()}] EXIT {symbol} @ {current_price:.2f} ({reason})"
                    )
                except Exception as e:
                    print(
                        f"[{now.isoformat()}] EXIT failed for {symbol} ({reason}): {e}"
                    )
                state.pop(symbol, None)

        # 4) Entry logic (momentum uptrend)
        open_positions = _get_open_positions_by_symbol(trading)
        for symbol in universe:
            if symbol in open_positions:
                continue
            window = price_windows[symbol]
            if len(window) < cfg.lookback_prices:
                continue
            if not _trend_up(window):
                print(f"[{now.isoformat()}] {symbol} not trending up; skipping")
                print(
                    f"[{now.isoformat()}]   Last observed prices: {', '.join([f'{p:.2f}' for p in window])}"
                )
                continue

            current_price = latest_prices.get(symbol)
            if current_price is None:
                continue

            try:
                _submit_market_buy_notional(trading, symbol, cfg.trade_notional_usd)
                state[symbol] = PositionState(entry_price=current_price, entry_time=now)
                print(
                    f"[{now.isoformat()}] ENTER {symbol} @ {current_price:.2f} notional=${cfg.trade_notional_usd:.2f}"
                )
            except Exception as e:
                print(f"[{now.isoformat()}] ENTER failed for {symbol}: {e}")

        time.sleep(cfg.poll_interval_seconds)


def main():
    _configure_unbuffered_output()
    cfg = StrategyConfig()
    run_strategy(cfg)


if __name__ == "__main__":
    main()
