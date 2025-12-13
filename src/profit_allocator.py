"""Helpers for recycling realized profits into accumulator coins and BNB."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Callable, Dict, Optional


PriceFetcher = Callable[[str], float]


@dataclass
class ProfitRecycleConfig:
    enabled: bool = False
    discount_allocation_pct: float = 0.10
    bnb_allocation_pct: float = 0.05
    min_order_quote: float = 11.0
    discount_symbol: str = ""
    bnb_symbol: str = "BNBUSDT"
    accumulation_filename: str = "profit_accumulation.json"
    rip_sell_fraction: float = 0.5

    @classmethod
    def from_dict(cls, raw: Dict[str, object]) -> "ProfitRecycleConfig":
        return cls(
            enabled=bool(raw.get("enabled", False)),
            discount_allocation_pct=float(raw.get("discount_allocation_pct", 0.10)),
            bnb_allocation_pct=float(raw.get("bnb_allocation_pct", 0.05)),
            min_order_quote=float(raw.get("min_order_quote", 11.0)),
            discount_symbol=str(raw.get("discount_symbol", "") or ""),
            bnb_symbol=str(raw.get("bnb_symbol", "BNBUSDT")),
            accumulation_filename=str(raw.get("accumulation_filename", "profit_accumulation.json")),
            rip_sell_fraction=float(raw.get("rip_sell_fraction", 0.5)),
        )


class ProfitAllocator:
    """Track profit splits for accumulating discount coins and BNB."""

    def __init__(self, config: ProfitRecycleConfig, savepoint_dir: Path):
        self.config = config
        self._lock = Lock()
        self.path = Path(savepoint_dir) / self.config.accumulation_filename
        self.state: Dict[str, object] = {
            "discount_pools": {},  # symbol -> {pending_quote, holdings_qty, quote_spent}
            "bnb_pending_quote": 0.0,
            "bnb_holdings_qty": 0.0,
            "fees_paid": {
                "totals": {"quote": 0.0, "bnb": 0.0, "other": {}, "by_strategy": {}},
                "per_pair": {},
            },
            "history": [],
            "last_updated": None,
        }
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    self.state.update(data)
        except (OSError, json.JSONDecodeError) as exc:
            logging.warning("Unable to load profit accumulation file %s: %s", self.path, exc)

    def _persist_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state["last_updated"] = datetime.now(timezone.utc).isoformat()
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(self.state, fh, ensure_ascii=False, indent=2, sort_keys=True)
        tmp.replace(self.path)

    def _record_history_locked(self, event: Dict[str, object]) -> None:
        history = self.state.setdefault("history", [])
        if isinstance(history, list):
            history.append(event)
            if len(history) > 500:
                del history[: len(history) - 500]

    def _pool_locked(self, symbol: str) -> Dict[str, float]:
        pools = self.state.setdefault("discount_pools", {})
        pool = pools.setdefault(symbol, {"pending_quote": 0.0, "holdings_qty": 0.0, "quote_spent": 0.0})
        return pool  # type: ignore[return-value]

    def _fees_bucket_locked(self, pair_symbol: str) -> Dict[str, object]:
        fees = self.state.setdefault("fees_paid", {})
        per_pair = fees.setdefault("per_pair", {})
        bucket = per_pair.setdefault(
            pair_symbol, {"quote": 0.0, "bnb": 0.0, "other": {}, "by_strategy": {}}
        )
        return bucket  # type: ignore[return-value]

    def _fees_strategy_bucket(self, bucket: Dict[str, object], source: str) -> Dict[str, object]:
        by_strategy = bucket.setdefault("by_strategy", {})
        return by_strategy.setdefault(source, {"quote": 0.0, "bnb": 0.0, "other": {}})

    def _bnb_asset(self, quote_asset: str) -> str:
        symbol = self.config.bnb_symbol.upper()
        quote = quote_asset.upper()
        if symbol.endswith(quote) and len(symbol) > len(quote):
            return symbol[: -len(quote)]
        if symbol.startswith("BNB"):
            return "BNB"
        return symbol

    def record_fees_from_fills(
        self, pair_symbol: str, fills, quote_asset: str, *, source: str = "ladder"
    ) -> None:
        if not fills:
            return
        quote = quote_asset.upper()
        bnb_asset = self._bnb_asset(quote)
        updated = False
        with self._lock:
            totals = self.state.setdefault("fees_paid", {}).setdefault(
                "totals", {"quote": 0.0, "bnb": 0.0, "other": {}, "by_strategy": {}}
            )
            for fill in fills:
                try:
                    commission = float(fill.get("commission", 0.0))
                    asset = str(fill.get("commissionAsset", "")).upper()
                except (TypeError, ValueError, AttributeError):
                    continue
                if commission <= 0 or not asset:
                    continue
                bucket = self._fees_bucket_locked(pair_symbol)
                strat_bucket = self._fees_strategy_bucket(bucket, source)
                totals_strat = self._fees_strategy_bucket(totals, source)
                if asset == quote:
                    bucket["quote"] = float(bucket.get("quote", 0.0)) + commission
                    totals["quote"] = float(totals.get("quote", 0.0)) + commission
                    strat_bucket["quote"] = float(strat_bucket.get("quote", 0.0)) + commission
                    totals_strat["quote"] = float(totals_strat.get("quote", 0.0)) + commission
                elif asset == bnb_asset:
                    bucket["bnb"] = float(bucket.get("bnb", 0.0)) + commission
                    totals["bnb"] = float(totals.get("bnb", 0.0)) + commission
                    strat_bucket["bnb"] = float(strat_bucket.get("bnb", 0.0)) + commission
                    totals_strat["bnb"] = float(totals_strat.get("bnb", 0.0)) + commission
                else:
                    other_bucket = bucket.setdefault("other", {})
                    total_other = totals.setdefault("other", {})
                    strat_other = strat_bucket.setdefault("other", {})
                    total_other_strat = totals_strat.setdefault("other", {})
                    other_bucket[asset] = float(other_bucket.get(asset, 0.0)) + commission
                    total_other[asset] = float(total_other.get(asset, 0.0)) + commission
                    strat_other[asset] = float(strat_other.get(asset, 0.0)) + commission
                    total_other_strat[asset] = float(total_other_strat.get(asset, 0.0)) + commission
                updated = True
            if updated:
                self._persist_locked()

    def allocate_profit(
        self,
        profit: float,
        *,
        pair_symbol: str,
        discount_symbol: str,
        price_lookup: PriceFetcher,
        client,
        dry_run: bool,
        activity_logger: Optional[Callable[[Dict[str, object]], None]] = None,
    ) -> None:
        if not self.config.enabled or profit <= 0:
            return
        target_symbol = discount_symbol.upper() if discount_symbol else pair_symbol.upper()
        now = datetime.now(timezone.utc).isoformat()
        discount_share = profit * max(self.config.discount_allocation_pct, 0.0)
        bnb_share = profit * max(self.config.bnb_allocation_pct, 0.0)
        with self._lock:
            pool = self._pool_locked(target_symbol)
            pool["pending_quote"] = float(pool.get("pending_quote", 0.0)) + discount_share
            self.state["bnb_pending_quote"] = float(self.state.get("bnb_pending_quote", 0.0)) + bnb_share
            self._record_history_locked(
                {
                    "ts": now,
                    "event": "ALLOCATE_PROFIT",
                    "pair": pair_symbol,
                    "discount_symbol": target_symbol,
                    "profit": profit,
                    "discount_share": discount_share,
                    "bnb_share": bnb_share,
                }
            )
            self._persist_locked()
        self.process_pending(target_symbol, price_lookup=price_lookup, client=client, dry_run=dry_run, activity_logger=activity_logger)

    def reserves_snapshot(self, discount_symbol: str) -> Dict[str, object]:
        symbol = discount_symbol.upper() if discount_symbol else ""
        with self._lock:
            pool = self._pool_locked(symbol) if symbol else {"pending_quote": 0.0, "holdings_qty": 0.0, "quote_spent": 0.0}
            snapshot = {
                "profit_reserve": {
                    "symbol": symbol or None,
                    "pending_quote": float(pool.get("pending_quote", 0.0)),
                    "holdings_qty": float(pool.get("holdings_qty", 0.0)),
                    "quote_spent": float(pool.get("quote_spent", 0.0)),
                },
                "bnb_reserve": {
                    "symbol": self.config.bnb_symbol,
                    "pending_quote": float(self.state.get("bnb_pending_quote", 0.0)),
                    "holdings_qty": float(self.state.get("bnb_holdings_qty", 0.0)),
                },
            }
        return snapshot

    def fees_snapshot(self, pair_symbol: str) -> Dict[str, object]:
        with self._lock:
            totals = self.state.get("fees_paid", {}).get("totals", {}) if isinstance(self.state.get("fees_paid"), dict) else {}
            per_pair = self.state.get("fees_paid", {}).get("per_pair", {}) if isinstance(self.state.get("fees_paid"), dict) else {}
            pair_bucket = per_pair.get(pair_symbol, {"quote": 0.0, "bnb": 0.0, "other": {}})
            snapshot = {
                "pair": {
                    "quote": float(pair_bucket.get("quote", 0.0)),
                    "bnb": float(pair_bucket.get("bnb", 0.0)),
                    "other": dict(pair_bucket.get("other", {})),
                    "by_strategy": {
                        name: {
                            "quote": float(bucket.get("quote", 0.0)),
                            "bnb": float(bucket.get("bnb", 0.0)),
                            "other": dict(bucket.get("other", {})),
                        }
                        for name, bucket in (pair_bucket.get("by_strategy") or {}).items()
                    },
                },
                "totals": {
                    "quote": float((totals or {}).get("quote", 0.0)),
                    "bnb": float((totals or {}).get("bnb", 0.0)),
                    "other": dict((totals or {}).get("other", {})),
                    "by_strategy": {
                        name: {
                            "quote": float(bucket.get("quote", 0.0)),
                            "bnb": float(bucket.get("bnb", 0.0)),
                            "other": dict(bucket.get("other", {})),
                        }
                        for name, bucket in (totals or {}).get("by_strategy", {}).items()
                    },
                },
            }
        return snapshot

    def _parse_executed_qty(self, resp: Dict[str, object]) -> float:
        if not isinstance(resp, dict):
            return 0.0
        try:
            qty = float(resp.get("executedQty") or resp.get("executed_qty") or 0.0)
            if qty > 0:
                return qty
        except (TypeError, ValueError):
            pass
        try:
            fills = resp.get("fills") or []
            total = 0.0
            for fill in fills:
                qty = float(fill.get("qty") or fill.get("quantity") or 0.0)
                total += qty
            return total
        except Exception:  # pragma: no cover - defensive
            return 0.0

    def _log_activity(self, activity_logger, payload: Dict[str, object]) -> None:
        if activity_logger:
            try:
                activity_logger(payload)
            except Exception:  # pragma: no cover - logging only
                logging.exception("Unable to write activity history for profit recycling")

    def _maybe_buy_discount(
        self,
        symbol: str,
        *,
        price_lookup: PriceFetcher,
        client,
        dry_run: bool,
        activity_logger,
    ) -> None:
        with self._lock:
            pool = self._pool_locked(symbol)
            pending = float(pool.get("pending_quote", 0.0))
            if pending < self.config.min_order_quote:
                return
            pool["pending_quote"] = 0.0
        price = price_lookup(symbol)
        if price <= 0 and not client:
            logging.info("Skipping profit recycle buy for %s: no price available in dry-run", symbol)
            with self._lock:
                pool = self._pool_locked(symbol)
                pool["pending_quote"] += pending
            return
        est_qty = pending / price if price > 0 else 0.0
        executed_qty = est_qty
        if client and not dry_run:
            try:
                resp = client.market_buy_quote(symbol, pending)
                executed_qty = self._parse_executed_qty(resp) or est_qty
            except Exception as exc:  # pylint: disable=broad-except
                logging.error("Failed to recycle profit into %s: %s", symbol, exc)
                with self._lock:
                    pool = self._pool_locked(symbol)
                    pool["pending_quote"] += pending
                    self._record_history_locked({"event": "DISCOUNT_BUY_FAILED", "symbol": symbol, "reason": str(exc), "ts": datetime.now(timezone.utc).isoformat()})
                    self._persist_locked()
                return
        with self._lock:
            pool = self._pool_locked(symbol)
            pool["holdings_qty"] = float(pool.get("holdings_qty", 0.0)) + executed_qty
            pool["quote_spent"] = float(pool.get("quote_spent", 0.0)) + pending
            event = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": "DISCOUNT_BUY",
                "symbol": symbol,
                "quote_used": pending,
                "est_price": price,
                "qty_added": executed_qty,
            }
            self._record_history_locked(event)
            self._persist_locked()
        self._log_activity(activity_logger, event)

    def _maybe_buy_bnb(self, *, price_lookup: PriceFetcher, client, dry_run: bool, activity_logger) -> None:
        with self._lock:
            pending = float(self.state.get("bnb_pending_quote", 0.0))
            if pending < self.config.min_order_quote:
                return
            self.state["bnb_pending_quote"] = 0.0
        price = price_lookup(self.config.bnb_symbol)
        if price <= 0 and not client:
            logging.info("Skipping BNB recycle: no price available in dry-run")
            with self._lock:
                self.state["bnb_pending_quote"] = float(self.state.get("bnb_pending_quote", 0.0)) + pending
            return
        est_qty = pending / price if price > 0 else 0.0
        executed_qty = est_qty
        if client and not dry_run:
            try:
                resp = client.market_buy_quote(self.config.bnb_symbol, pending)
                executed_qty = self._parse_executed_qty(resp) or est_qty
            except Exception as exc:  # pylint: disable=broad-except
                logging.error("Failed to recycle profit into BNB: %s", exc)
                with self._lock:
                    self.state["bnb_pending_quote"] = float(self.state.get("bnb_pending_quote", 0.0)) + pending
                    self._record_history_locked({"event": "BNB_BUY_FAILED", "reason": str(exc), "ts": datetime.now(timezone.utc).isoformat()})
                    self._persist_locked()
                return
        with self._lock:
            self.state["bnb_holdings_qty"] = float(self.state.get("bnb_holdings_qty", 0.0)) + executed_qty
            event = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": "BNB_BUY",
                "symbol": self.config.bnb_symbol,
                "quote_used": pending,
                "est_price": price,
                "qty_added": executed_qty,
            }
            self._record_history_locked(event)
            self._persist_locked()
        self._log_activity(activity_logger, event)

    def process_pending(
        self,
        discount_symbol: str,
        *,
        price_lookup: PriceFetcher,
        client,
        dry_run: bool,
        activity_logger: Optional[Callable[[Dict[str, object]], None]] = None,
    ) -> None:
        if not self.config.enabled:
            return
        symbol = discount_symbol.upper() if discount_symbol else ""
        if symbol:
            self._maybe_buy_discount(symbol, price_lookup=price_lookup, client=client, dry_run=dry_run, activity_logger=activity_logger)
        self._maybe_buy_bnb(price_lookup=price_lookup, client=client, dry_run=dry_run, activity_logger=activity_logger)

    def sell_on_rip(
        self,
        discount_symbol: str,
        *,
        price_lookup: PriceFetcher,
        client,
        dry_run: bool,
        activity_logger: Optional[Callable[[Dict[str, object]], None]] = None,
        hinted_price: Optional[float] = None,
    ) -> None:
        if not self.config.enabled:
            return
        symbol = discount_symbol.upper()
        with self._lock:
            pool = self._pool_locked(symbol)
            holdings = float(pool.get("holdings_qty", 0.0))
        if holdings <= 0:
            return
        sell_qty = holdings * max(min(self.config.rip_sell_fraction, 1.0), 0.0)
        price = hinted_price or price_lookup(symbol)
        if price <= 0:
            return
        if sell_qty * price < self.config.min_order_quote:
            logging.info("Skipping rip sell for %s: not enough value (%.2f)", symbol, sell_qty * price)
            return
        executed_qty = sell_qty
        if client and not dry_run:
            try:
                resp = client.market_sell(symbol, sell_qty)
                executed_qty = self._parse_executed_qty(resp) or sell_qty
            except Exception as exc:  # pylint: disable=broad-except
                logging.error("Failed to sell stash on rip for %s: %s", symbol, exc)
                return
        proceeds = price * executed_qty
        with self._lock:
            pool = self._pool_locked(symbol)
            pool["holdings_qty"] = max(float(pool.get("holdings_qty", 0.0)) - executed_qty, 0.0)
            event = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": "RIP_SELL",
                "symbol": symbol,
                "qty_sold": executed_qty,
                "price_hint": price,
                "proceeds_est": proceeds,
            }
            self._record_history_locked(event)
            self._persist_locked()
        self._log_activity(activity_logger, event)