from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from .utils.config import load_app_config
from .utils.logging import configure_logging


@dataclass
class RebalanceConfig:
    frequency: str  # "monthly" or "daily"
    as_of: Optional[pd.Timestamp]
    top: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive Rebalance Helper")
    p.add_argument("--strategy", default=str(Path(__file__).resolve().parents[1] / "config" / "strategy.yml"), help="Path to strategy.yml")
    p.add_argument("--frequency", choices=["monthly", "daily"], default="monthly", help="Weights frequency to use")
    p.add_argument("--as-of", default="latest", help="Date to use (YYYY-MM-DD) or 'latest'")
    p.add_argument("--top", type=int, default=20, help="Top N stocks to display in summary (printing only)")
    return p.parse_args()


def _parse_date(s: Optional[str]) -> Optional[pd.Timestamp]:
    if s is None:
        return None
    if isinstance(s, str) and s.lower() == "latest":
        return None
    return pd.to_datetime(s)


def _weights_globs(freq: str) -> Tuple[str, str]:
    if freq == "daily":
        return ("sector_weights_daily_*.csv", "stock_weights_daily_*.csv")
    return ("sector_weights_monthly_*.csv", "stock_weights_monthly_*.csv")


def _pick_asof_index(idx: pd.DatetimeIndex, as_of: Optional[pd.Timestamp]) -> Optional[pd.Timestamp]:
    if idx.empty:
        return None
    if as_of is None:
        return idx[-1]
    as_of = pd.Timestamp(as_of)
    loc = idx.get_indexer([as_of], method="pad")[0]
    if loc == -1:
        return None
    return idx[loc]


def _format_pct(x: float) -> str:
    return f"{x*100.0:6.2f}%"


def _format_ccy(x: float) -> str:
    return f"${x:,.2f}"


def _print_series(logger, header: str, s: pd.Series, top: Optional[int] = None, cash: float = 0.0) -> None:
    logger.info(header)
    if s.empty and cash <= 0.0:
        logger.info("(empty)")
        return
    if top is not None and top > 0:
        s = s.head(top)
    for i, (k, v) in enumerate(s.items(), start=1):
        logger.info("%2d) %-24s %s", i, str(k), _format_pct(float(v)))
    if cash > 1e-12:
        logger.info("    %-24s %s", "Cash", _format_pct(float(cash)))


def _compute_cash_weight(row: pd.Series) -> float:
    total = float(row.fillna(0.0).sum()) if len(row) else 0.0
    return max(0.0, 1.0 - total)


def _load_weights(cfg, frequency: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    weights_dir = (cfg.output_root_path / "weights").resolve()
    sector_glob, stock_glob = _weights_globs(frequency)

    # Sector weights
    sec_files = sorted(weights_dir.glob(sector_glob))
    sec_df = pd.DataFrame()
    if sec_files:
        sec_df = pd.read_csv(sec_files[-1], index_col=0)
        if not sec_df.empty:
            try:
                sec_df.index = pd.to_datetime(sec_df.index)
            except Exception:
                pass

    # Stock weights
    stk_files = sorted(weights_dir.glob(stock_glob))
    stk_df = pd.DataFrame()
    if stk_files:
        stk_df = pd.read_csv(stk_files[-1], index_col=0)
        if not stk_df.empty:
            try:
                stk_df.index = pd.to_datetime(stk_df.index)
            except Exception:
                pass

    return sec_df, stk_df


def _prompt_positions(logger) -> Tuple[Dict[str, float], float]:
    logger.info("Enter current stock positions one per line as: TICKER amount_in_dollars")
    logger.info("Type 'done' on a new line to finish. Press Enter on an empty line to skip.")
    positions: Dict[str, float] = {}
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            # empty line, continue prompting
            continue
        lc = line.lower()
        if lc in {"done", "exit", "quit"}:
            break
        parts = [p for p in line.replace(",", " ").split() if p]
        if len(parts) < 2:
            print("  Expect: TICKER amount (e.g., AAPL 12345). Try again or type 'done'.")
            continue
        ticker, amt_str = parts[0].upper(), parts[1]
        try:
            amt = float(amt_str)
        except Exception:
            print("  Amount must be a number. Try again.")
            continue
        positions[ticker] = positions.get(ticker, 0.0) + amt
    # Cash prompt
    cash = 0.0
    try:
        line = input("Enter current cash amount (blank for 0): ").strip()
        if line:
            cash = float(line)
    except Exception:
        print("  Could not parse cash amount; using 0.")
        cash = 0.0
    return positions, cash


def main() -> int:
    args = parse_args()
    cfg = load_app_config(Path(args.strategy).resolve())
    logger = configure_logging(cfg.output_root_path, level=cfg.runtime.log_level, log_to_file=cfg.runtime.save.get("logs", True))

    frequency = args.frequency
    as_of = _parse_date(args.as_of)
    top = int(args.top)

    # Load latest weights for chosen frequency
    sec_df, stk_df = _load_weights(cfg, frequency)
    if stk_df.empty:
        logger.error("No %s stock weights available; compute weights first", frequency)
        return 1

    # Determine as-of date
    dt = _pick_asof_index(stk_df.index, as_of)
    if dt is None:
        logger.error("No weight date on/before %s", args.as_of)
        return 1

    # Extract rows
    row_stk = stk_df.loc[dt].fillna(0.0)
    row_stk = row_stk[row_stk > 0].sort_values(ascending=False)
    row_sec = pd.Series(dtype=float)
    if not sec_df.empty:
        if dt in sec_df.index:
            row_sec = sec_df.loc[dt].fillna(0.0)
        else:
            # pick last sector date <= dt
            sec_dt = _pick_asof_index(sec_df.index, dt)
            if sec_dt is not None:
                row_sec = sec_df.loc[sec_dt].fillna(0.0)
        if not row_sec.empty:
            row_sec = row_sec[row_sec > 0].sort_values(ascending=False)

    cash_w = _compute_cash_weight(row_stk)

    # Summary header
    logger.info("=== Rebalance Helper ===")
    logger.info("Using %s weights as of %s", frequency.upper(), dt.date())

    # Print sector & stock summaries
    if not row_sec.empty:
        _print_series(logger, "--- Sector allocation ---", row_sec, None, 0.0)
    _print_series(logger, f"--- Stock allocation (top {top}) ---", row_stk, top, cash_w)

    # Prompt current positions
    positions, cash = _prompt_positions(logger)

    # Compute totals
    current_equity = float(sum(positions.values())) + float(cash)
    if current_equity <= 0:
        logger.warning("Current equity computed as 0; cannot size targets. Exiting.")
        return 1

    # Compute targets
    targets: Dict[str, float] = {}
    for tkr, w in row_stk.items():
        targets[tkr] = float(w) * current_equity
    target_cash = float(cash_w) * current_equity

    # Execution deltas
    union = set(positions.keys()) | set(targets.keys())
    buys: Dict[str, float] = {}
    sells: Dict[str, float] = {}
    for t in sorted(union):
        cur = float(positions.get(t, 0.0))
        tgt = float(targets.get(t, 0.0))
        delta = tgt - cur
        if delta > 1e-9:
            buys[t] = delta
        elif delta < -1e-9:
            sells[t] = -delta

    total_buys = sum(buys.values())
    total_sells = sum(sells.values())
    end_cash = float(cash) + total_sells - total_buys
    cash_gap = end_cash - target_cash

    # Output plan
    logger.info("=== Execution Plan ===")
    logger.info("Current equity: %s | Target cash: %s | Projected end cash: %s | Cash diff: %s",
                _format_ccy(current_equity), _format_ccy(target_cash), _format_ccy(end_cash), _format_ccy(cash_gap))

    if buys:
        logger.info("-- Buys (dollar amounts) --")
        for t, v in sorted(buys.items(), key=lambda kv: -kv[1]):
            logger.info("BUY  %-10s %s", t, _format_ccy(v))
    else:
        logger.info("-- No buys required --")

    if sells:
        logger.info("-- Sells (dollar amounts) --")
        for t, v in sorted(sells.items(), key=lambda kv: -kv[1]):
            logger.info("SELL %-10s %s", t, _format_ccy(v))
    else:
        logger.info("-- No sells required --")

    # Also print target positions for convenience
    logger.info("=== Target Positions (dollars) ===")
    for t, v in row_stk.items():
        logger.info("TGT  %-10s %s  (w=%s)", t, _format_ccy(float(v) * current_equity), _format_pct(float(v)))
    if cash_w > 0:
        logger.info("TGT  %-10s %s  (w=%s)", "CASH", _format_ccy(target_cash), _format_pct(float(cash_w)))

    # Footer note
    if abs(cash_gap) > 0.01:
        logger.info("Note: Cash diff is non-zero due to rounding or missing tickers; adjust trades as needed to hit target cash.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
