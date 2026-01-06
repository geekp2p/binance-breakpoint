"""Utility to generate time-boxed overrides for a stuck symbol.

This script helps apply the reversible override cycle described in the
optimization notes without manually editing the entire config. It prints a YAML
snippet for the requested symbol and stage, optionally merging it into an
existing YAML config when ``--write`` is provided.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from typing import Any, Dict

import yaml

# Baseline override (first 2–3 day nudge).
BASE_OVERRIDE: Dict[str, Any] = {
    "buy_ladder": {
        "d_buy": 0.02,
        "m_buy": 1.35,
        "gap_factor": 1.08,
        "max_quote_per_leg": 0.08,  # fraction of b_alloc; replace with absolute if preferred
    },
    "adaptive_ladder": {
        "min_d_buy": 0.02,
        "max_d_buy": 0.09,
    },
    "micro_oscillation": {
        "max_band_pct": 0.012,
        "entry_band_pct": 0.12,
        "min_swing_pct": 0.0008,
        "take_profit_pct": 0.004,
        "cooldown_bars": 5,
    },
    "profit_trail": {
        "p_min": 0.025,
        "s1": 0.012,
        "p_lock_base": 0.007,
        "p_lock_max": 0.03,
        "no_loss_epsilon": 0.0015,
    },
    "time_caps": {
        "T_idle_max_minutes": 70,
        "p_idle": 0.006,
        "p_exit_min": 0.01,
    },
    "features": {
        "scalp_mode": {
            "order_pct_allocation": 0.45,
            "cooldown_bars": 2,
        },
    },
    "profit_recycling": {
        "enabled": True,
        "discount_allocation_pct": 0.05,
        "bnb_allocation_pct": 0.03,
        "min_order_quote": 20,
    },
}


@dataclass(frozen=True)
class StageAdjustment:
    label: str
    note: str
    patch: Dict[str, Any]


# Second stage (7–14 day follow-up) gently widens buy spacing and micro bands.
FOLLOW_UP_ADJUSTMENT = StageAdjustment(
    label="follow-up",
    note=(
        "Applied after 7–14 days of inactivity. Widens spacing a bit further and "
        "loosens micro entry bands. Revert once the pair resumes filling."
    ),
    patch={
        "buy_ladder": {
            "d_buy": 0.025,
        },
        "adaptive_ladder": {
            "min_d_buy": 0.025,
        },
        "micro_oscillation": {
            "entry_band_pct": 0.14,
            "max_band_pct": 0.014,
        },
    },
)


def build_override(stage: str) -> Dict[str, Any]:
    """Return a merged override for the requested stage.

    ``stage`` can be ``baseline`` (first 2–3 day nudge) or ``follow-up``
    (7–14 day escalation). The follow-up merges the baseline with a mild patch.
    """

    stage = stage.lower()
    override = copy.deepcopy(BASE_OVERRIDE)
    if stage == "baseline":
        return override
    if stage == FOLLOW_UP_ADJUSTMENT.label:
        return _deep_merge(override, FOLLOW_UP_ADJUSTMENT.patch)
    raise ValueError("stage must be 'baseline' or 'follow-up'")


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def format_snippet(symbol: str, override: Dict[str, Any]) -> str:
    return yaml.safe_dump({symbol: override}, sort_keys=False)


def merge_into_config(config_path: str, symbol: str, override: Dict[str, Any]) -> None:
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    symbols_cfg = config.setdefault("symbols", {})
    symbols_cfg[symbol] = _deep_merge(symbols_cfg.get(symbol, {}) or {}, override)
    with open(config_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbol", help="Symbol to override, e.g., DCRUSDT")
    parser.add_argument(
        "--stage",
        choices=["baseline", FOLLOW_UP_ADJUSTMENT.label],
        default="baseline",
        help=(
            "baseline = first 2–3 day nudge; follow-up = 7–14 day escalation "
            "if the pair remains idle"
        ),
    )
    parser.add_argument(
        "--config",
        help="Path to config YAML to patch (only used with --write)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the override directly into the provided config file",
    )
    args = parser.parse_args()

    override = build_override(args.stage)
    snippet = format_snippet(args.symbol, override)
    print("# Override snippet (stage: %s)\n%s" % (args.stage, snippet))

    if args.write:
        if not args.config:
            raise SystemExit("--write requires --config")
        merge_into_config(args.config, args.symbol, override)
        print(f"Updated {args.config} with overrides for {args.symbol} ({args.stage}).")


if __name__ == "__main__":
    main()
