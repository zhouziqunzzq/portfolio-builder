from __future__ import annotations

import asyncio
import os
import time
from typing import Any, List, Optional

from events.event_bus import EventBus
from events.events import (
    AccountSnapshotEvent,
    BrokerAccount,
    BrokerPosition,
    RebalancePlanRequestEvent,
)
from runtime_manager import RuntimeManager

from .base_eml import BaseEML
from .config import EMLConfig


class AlpacaEMLService(BaseEML):
    """Alpaca Execution Market Link (EML).

    Current scope:
    - Periodically fetch account + (optionally) positions from Alpaca
    - Publish an `AccountSnapshotEvent` to the event bus

    Future scope:
    - Translate `RebalancePlanRequestEvent` into broker orders
    - Publish order/fill updates to the bus
    """

    def __init__(
        self,
        bus: "EventBus",
        rm: Optional["RuntimeManager"] = None,
        *,
        config: Optional[EMLConfig] = None,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
        paper: Optional[bool] = None,
        name: str = "AlpacaEML",
    ):
        super().__init__(bus=bus, name=name)

        if config is None:
            self.log.warning("No EMLConfig provided; using default configuration")
            config = EMLConfig()
        self._validate_config(config)
        self.config = config

        if self.config.polling_interval_secs <= 0:
            raise ValueError("polling_interval_secs must be > 0")
        self._poll_interval_seconds = float(self.config.polling_interval_secs)

        self.rm = rm
        if self.rm is not None:
            # Register self to RuntimeManager for lifecycle management.
            self.rm.set("eml", self)
            self.rm.set("alpaca_eml", self)  # alias

        self._api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self._secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        self._base_url = base_url or os.environ.get(
            "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
        )

        if paper is None:
            env_paper = os.environ.get("ALPACA_PAPER")
            if env_paper is None:
                self._paper = True
            else:
                self._paper = env_paper.strip().lower() in {"1", "true", "yes", "y"}
        else:
            self._paper = bool(paper)

        self._trading = self._build_trading_client()
        self.log.info(
            "Initialized AlpacaEMLService (paper=%s, base_url=%s)",
            self._paper,
            self._base_url,
        )

        # TODO: Add order management state here as needed.

    def _build_trading_client(self):
        if not self._api_key or not self._secret_key:
            raise RuntimeError(
                "Alpaca credentials missing. Set ALPACA_API_KEY and ALPACA_SECRET_KEY (or pass api_key/secret_key)."
            )

        try:
            from alpaca.trading.client import TradingClient
        except Exception as e:
            raise RuntimeError(
                "Missing dependency 'alpaca-py'. Install with `pip install alpaca-py`."
            ) from e

        return TradingClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
            paper=self._paper,
            url_override=self._base_url,
        )

    async def _run_loop(self) -> None:
        while self._running:
            try:
                account = await asyncio.to_thread(self._get_account)
                positions: List[BrokerPosition] = []
                if self.config.include_positions:
                    positions = await asyncio.to_thread(self._list_positions)

                event = AccountSnapshotEvent(
                    ts=time.time(),
                    source=self.name,
                    account=account,
                    positions=positions,
                )
                self.log.debug(
                    "Fetched account snapshot: account=%s positions=%s",
                    account,
                    positions,
                )
                await self.emit_account_event(event)

                await asyncio.sleep(self._poll_interval_seconds)
            except asyncio.CancelledError:
                raise
            except Exception:
                # Keep running; transient API/network issues are expected.
                self.log.exception("Error in AlpacaEMLService main loop")
                await asyncio.sleep(min(30.0, max(1.0, self._poll_interval_seconds)))

    async def execute_rebalance_plan(self, event: RebalancePlanRequestEvent) -> None:
        # Not implemented yet.
        # For now we just log; future: translate weights into orders + publish order updates.
        self.log.warning(
            "Rebalance execution not implemented yet; ignoring RebalancePlanRequestEvent: rebalance_id=%s weights=%s",
            event.rebalance_id,
            event.weights,
        )

    @staticmethod
    def _to_float(v: Any) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            try:
                return float(s)
            except ValueError:
                return None
        return None

    def _get_account(self) -> BrokerAccount:
        acct = self._trading.get_account()
        raw = getattr(acct, "_raw", None)
        if raw is not None:
            return BrokerAccount(
                id=raw.get("id"),
                status=raw.get("status"),
                cash=self._to_float(raw.get("cash")),
                buying_power=self._to_float(raw.get("buying_power")),
                portfolio_value=self._to_float(raw.get("portfolio_value")),
                equity=self._to_float(raw.get("equity")),
                last_equity=self._to_float(raw.get("last_equity")),
                adj_equity=self._get_equity_adj(self._to_float(raw.get("equity"))),
            )

        return BrokerAccount(
            id=getattr(acct, "id", None),
            status=getattr(acct, "status", None),
            cash=self._to_float(getattr(acct, "cash", None)),
            buying_power=self._to_float(getattr(acct, "buying_power", None)),
            portfolio_value=self._to_float(getattr(acct, "portfolio_value", None)),
            equity=self._to_float(getattr(acct, "equity", None)),
            last_equity=self._to_float(getattr(acct, "last_equity", None)),
            adj_equity=self._get_equity_adj(
                self._to_float(getattr(acct, "equity", None))
            ),
        )

    def _list_positions(self) -> List[BrokerPosition]:
        try:
            pos_list = self._trading.get_all_positions()
        except Exception:
            try:
                pos_list = self._trading.get_positions()
            except Exception:
                pos_list = []

        out: List[BrokerPosition] = []
        for p in pos_list:
            raw = getattr(p, "_raw", None)
            if raw is not None:
                symbol = raw.get("symbol")
                if not symbol:
                    continue
                out.append(
                    BrokerPosition(
                        symbol=str(symbol),
                        qty=self._to_float(raw.get("qty")),
                        market_value=self._to_float(raw.get("market_value")),
                        avg_entry_price=self._to_float(raw.get("avg_entry_price")),
                        side=raw.get("side"),
                        unrealized_pl=self._to_float(raw.get("unrealized_pl")),
                    )
                )
                continue

            symbol = getattr(p, "symbol", None)
            if not symbol:
                continue
            out.append(
                BrokerPosition(
                    symbol=str(symbol),
                    qty=self._to_float(getattr(p, "qty", None)),
                    market_value=self._to_float(getattr(p, "market_value", None)),
                    avg_entry_price=self._to_float(getattr(p, "avg_entry_price", None)),
                    side=getattr(p, "side", None),
                    unrealized_pl=self._to_float(getattr(p, "unrealized_pl", None)),
                )
            )

        return out

    @staticmethod
    def _validate_config(config: EMLConfig) -> None:
        if config.cash_buffer_pct is not None and config.cash_buffer_abs is not None:
            raise ValueError(
                "EMLConfig: cash_buffer_pct and cash_buffer_abs are mutually exclusive; only one may be set."
            )

    def _get_equity_adj(self, equity_abs: Optional[float]) -> Optional[float]:
        """Compute adjusted equity after applying cash buffer settings.

        Returns None if equity_abs is None.
        """
        if equity_abs is None:
            return None

        if self.config.cash_buffer_pct is not None:
            buffer_amt = equity_abs * self.config.cash_buffer_pct
            return max(0.0, equity_abs - buffer_amt)

        if self.config.cash_buffer_abs is not None:
            buffer_amt = self.config.cash_buffer_abs
            return max(0.0, equity_abs - buffer_amt)

        return equity_abs
