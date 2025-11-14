"""Utilities for persisting and restoring live trading state."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .strategy import StrategyState


DEFAULT_SAVEPOINT_DIR = Path("savepoint")
SAVEPOINT_VERSION = 1


def _ensure_dir(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)


def _to_iso(value: Optional[Any]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc).isoformat()
        return value.isoformat()
    try:
        import pandas as pd  # type: ignore

        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime().isoformat()
    except Exception:  # pragma: no cover - pandas optional
        pass
    return str(value)


def _from_iso(value: Optional[str]) -> Optional[datetime]:
    if value in (None, ""):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except Exception:  # pragma: no cover - fallback
            logging.warning("Unable to parse timestamp %s from savepoint", value)
            return None


def _serialise_events(events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for ev in events:
        out: Dict[str, Any] = {}
        for key, value in ev.items():
            if key == "ts":
                out[key] = _to_iso(value)
            else:
                out[key] = value
        payload.append(out)
    return payload


def _deserialise_events(raw: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for ev in raw:
        out: Dict[str, Any] = {}
        for key, value in ev.items():
            if key == "ts" and isinstance(value, str):
                out[key] = _from_iso(value)
            else:
                out[key] = value
        events.append(out)
    return events


def state_to_payload(state: StrategyState) -> Dict[str, Any]:
    return {
        "round_active": state.round_active,
        "phase": state.phase,
        "P0": state.P0,
        "ladder_prices": list(state.ladder_prices),
        "ladder_amounts_quote": list(state.ladder_amounts_quote),
        "ladder_next_idx": state.ladder_next_idx,
        "Q": state.Q,
        "C": state.C,
        "TP_base": state.TP_base,
        "H": state.H,
        "stage": state.stage,
        "cumulative_s": state.cumulative_s,
        "tau_cur": state.tau_cur,
        "p_lock_cur": state.p_lock_cur,
        "round_start_ts": _to_iso(state.round_start_ts),
        "last_high_ts": _to_iso(state.last_high_ts),
        "next_window_minutes": state.next_window_minutes,
        "idle_checked": state.idle_checked,
        "btd_armed": state.btd_armed,
        "btd_last_bottom": state.btd_last_bottom,
        "btd_last_arm_ts": _to_iso(state.btd_last_arm_ts),
        "btd_cooldown_until": _to_iso(state.btd_cooldown_until),
        "sah_armed": state.sah_armed,
        "sah_last_top": state.sah_last_top,
        "sah_last_arm_ts": _to_iso(state.sah_last_arm_ts),
        "sah_cooldown_until": _to_iso(state.sah_cooldown_until),
    }


def apply_payload_to_state(state: StrategyState, payload: Dict[str, Any]) -> None:
    state.round_active = bool(payload.get("round_active", state.round_active))
    state.phase = payload.get("phase", state.phase)
    state.P0 = payload.get("P0", state.P0)
    state.ladder_prices = list(payload.get("ladder_prices", state.ladder_prices))
    state.ladder_amounts_quote = list(payload.get("ladder_amounts_quote", state.ladder_amounts_quote))
    state.ladder_next_idx = int(payload.get("ladder_next_idx", state.ladder_next_idx))
    state.Q = float(payload.get("Q", state.Q))
    state.C = float(payload.get("C", state.C))
    state.TP_base = payload.get("TP_base", state.TP_base)
    state.H = payload.get("H", state.H)
    state.stage = int(payload.get("stage", state.stage))
    state.cumulative_s = float(payload.get("cumulative_s", state.cumulative_s))
    state.tau_cur = float(payload.get("tau_cur", state.tau_cur))
    state.p_lock_cur = float(payload.get("p_lock_cur", state.p_lock_cur))
    state.round_start_ts = _from_iso(payload.get("round_start_ts")) or state.round_start_ts
    state.last_high_ts = _from_iso(payload.get("last_high_ts")) or state.last_high_ts
    nwm = payload.get("next_window_minutes", state.next_window_minutes)
    state.next_window_minutes = float(nwm) if nwm is not None else None
    state.idle_checked = bool(payload.get("idle_checked", state.idle_checked))
    state.btd_armed = bool(payload.get("btd_armed", state.btd_armed))
    state.btd_last_bottom = payload.get("btd_last_bottom", state.btd_last_bottom)
    state.btd_last_arm_ts = _from_iso(payload.get("btd_last_arm_ts")) or state.btd_last_arm_ts
    state.btd_cooldown_until = _from_iso(payload.get("btd_cooldown_until")) or state.btd_cooldown_until
    state.sah_armed = bool(payload.get("sah_armed", state.sah_armed))
    state.sah_last_top = payload.get("sah_last_top", state.sah_last_top)
    state.sah_last_arm_ts = _from_iso(payload.get("sah_last_arm_ts")) or state.sah_last_arm_ts
    state.sah_cooldown_until = _from_iso(payload.get("sah_cooldown_until")) or state.sah_cooldown_until


def load_savepoint(directory: Path, symbol: str) -> Optional[Dict[str, Any]]:
    save_file = directory / f"{symbol.upper()}.json"
    if not save_file.exists():
        return None
    try:
        with save_file.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except json.JSONDecodeError:
        logging.error("Savepoint %s is corrupted", save_file)
        return None
    if payload.get("version") != SAVEPOINT_VERSION:
        logging.warning("Savepoint version mismatch: expected %s got %s", SAVEPOINT_VERSION, payload.get("version"))
    payload["event_log"] = _deserialise_events(payload.get("event_log", []))
    payload["activity_history"] = _deserialise_events(payload.get("activity_history", []))
    return payload


def write_savepoint(
    directory: Path,
    symbol: str,
    state: StrategyState,
    *,
    last_open_time: int,
    event_log: Iterable[Dict[str, Any]],
    activity_history: Iterable[Dict[str, Any]],
    realized_pnl_total: float,
    latest_price: float,
    status: Dict[str, Any],
) -> None:
    _ensure_dir(directory)
    save_file = directory / f"{symbol.upper()}.json"
    payload = {
        "version": SAVEPOINT_VERSION,
        "symbol": symbol.upper(),
        "saved_at": _to_iso(datetime.utcnow()),
        "state": state_to_payload(state),
        "last_open_time": last_open_time,
        "event_log": _serialise_events(event_log),
        "activity_history": _serialise_events(activity_history),
        "realized_pnl_total": realized_pnl_total,
        "latest_price": latest_price,
        "status": status,
    }
    tmp_file = save_file.with_suffix(".tmp")
    with tmp_file.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
    tmp_file.replace(save_file)


__all__ = [
    "DEFAULT_SAVEPOINT_DIR",
    "apply_payload_to_state",
    "load_savepoint",
    "state_to_payload",
    "write_savepoint",
]