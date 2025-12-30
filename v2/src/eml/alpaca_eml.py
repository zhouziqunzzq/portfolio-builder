from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List, Optional

from events.event_bus import EventBus
from events.events import AccountSnapshotEvent, RebalancePlanRequestEvent
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
                account = await asyncio.to_thread(self._get_account_info)
                positions: List[Dict[str, Any]] = []
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

    def _get_account_info(self) -> Dict[str, Any]:
        acct = self._trading.get_account()
        raw = getattr(acct, "_raw", None)
        if raw is not None:
            return raw

        return {
            "id": getattr(acct, "id", None),
            "status": getattr(acct, "status", None),
            "cash": getattr(acct, "cash", None),
            "buying_power": getattr(acct, "buying_power", None),
            "portfolio_value": getattr(acct, "portfolio_value", None),
            "equity": getattr(acct, "equity", None),
            "last_equity": getattr(acct, "last_equity", None),
        }

    def _list_positions(self) -> List[Dict[str, Any]]:
        try:
            pos_list = self._trading.get_all_positions()
        except Exception:
            try:
                pos_list = self._trading.get_positions()
            except Exception:
                pos_list = []

        out: List[Dict[str, Any]] = []
        for p in pos_list:
            raw = getattr(p, "_raw", None)
            if raw is not None:
                out.append(raw)
                continue

            out.append(
                {
                    "symbol": getattr(p, "symbol", None),
                    "qty": getattr(p, "qty", None),
                    "market_value": getattr(p, "market_value", None),
                    "avg_entry_price": getattr(p, "avg_entry_price", None),
                    "side": getattr(p, "side", None),
                    "unrealized_pl": getattr(p, "unrealized_pl", None),
                }
            )

        return out
