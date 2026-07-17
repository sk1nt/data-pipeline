"""Helpers for deriving compact GEX wall candidate payloads."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _bounded_pct(value: Any) -> Optional[float]:
    """Normalize an incoming wall percentage to the monitor's 0-100 range."""
    if value is None:
        return None
    try:
        numeric = float(value)
        return min(100.0, max(0.0, numeric)) if math.isfinite(numeric) else None
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    except (TypeError, ValueError):
        return None


def parse_gex_strikes(raw: Any) -> List[Tuple[float, float]]:
    """Normalize a strike ladder into ``[(strike, gamma), ...]`` pairs."""
    if not isinstance(raw, list):
        return []
    strikes: List[Tuple[float, float]] = []
    for entry in raw:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            try:
                strike = float(entry[0])
                gamma = float(entry[1])
                if math.isfinite(strike) and math.isfinite(gamma) and gamma != 0.0:
                    strikes.append((strike, gamma))
            except (TypeError, ValueError):
                continue
        elif isinstance(entry, dict):
            try:
                strike = float(entry.get("strike"))
                # Dict payloads may name the volume field explicitly. Never
                # fall back to oi_gamma for volume-based wall candidates.
                volume_gamma = entry.get(
                    "volume_gamma", entry.get("gamma")
                )
                gamma = float(volume_gamma)
                if math.isfinite(strike) and math.isfinite(gamma) and gamma != 0.0:
                    strikes.append((strike, gamma))
            except (TypeError, ValueError):
                continue
    return strikes


def summarize_wall_candidates(
    major_strike: Any,
    strikes: Sequence[Tuple[float, float]],
    *,
    prefer_positive: bool,
) -> List[Dict[str, Any]]:
    """Return the major wall plus the next two candidates.

    The caller chooses the side via ``prefer_positive``. The returned list
    contains at most two entries beyond the major wall, each with strike, signed
    gamma value, and percent of the major wall magnitude.
    """
    filtered: List[Tuple[float, float]] = []
    for strike, gamma in strikes:
        if not math.isfinite(gamma):
            continue
        if prefer_positive and gamma <= 0:
            continue
        if not prefer_positive and gamma >= 0:
            continue
        filtered.append((strike, gamma))
    if not filtered:
        return []

    filtered.sort(key=lambda pair: abs(pair[1]), reverse=True)
    major: Optional[Tuple[float, float]] = None
    if isinstance(major_strike, (int, float)):
        for strike, gamma in filtered:
            if abs(strike - major_strike) <= 0.51:
                major = (strike, gamma)
                break
    if major is None:
        major = filtered[0]

    # The percentage is relative to the strongest wall on this side.  The
    # provider's major-wall strike can be stale/misaligned with the ladder;
    # using its gamma as the denominator can therefore produce percentages
    # above 100% in the monitor.
    reference_gamma = max(filtered, key=lambda pair: abs(pair[1]))[1]

    entries: List[Dict[str, Any]] = []
    for strike, gamma in filtered:
        if abs(strike - major[0]) <= 0.51:
            continue
        entries.append(
            {
                "strike": strike,
                "value": gamma,
                "pct": _bounded_pct(
                    abs(gamma) / abs(reference_gamma) * 100
                    if reference_gamma
                    else None
                ),
            }
        )
        if len(entries) >= 2:
            break
    return entries


def _major_wall_gamma_fields(
    snapshot: Dict[str, Any], *, prefer_positive: bool
) -> Dict[str, Any]:
    major_key = "major_pos_vol" if prefer_positive else "major_neg_vol"
    gamma_key = "major_pos_vol_gamma" if prefer_positive else "major_neg_vol_gamma"

    strikes = parse_gex_strikes(snapshot.get("strikes"))
    side_strikes = [
        (strike, gamma)
        for strike, gamma in strikes
        if (gamma > 0 if prefer_positive else gamma < 0)
    ]

    wall_gamma = _coerce_float(snapshot.get(gamma_key))
    wall_strike = _coerce_float(snapshot.get(major_key))
    if wall_gamma is None and wall_strike is not None:
        for strike, gamma in side_strikes:
            if abs(strike - wall_strike) <= 0.51:
                wall_gamma = gamma
                break
    if wall_gamma is None and side_strikes:
        wall_gamma = max(side_strikes, key=lambda pair: abs(pair[1]))[1]

    fields: Dict[str, Any] = {}
    if wall_gamma is not None:
        fields[gamma_key] = wall_gamma
    return fields


def build_compact_wall_fields(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Populate compact wall fields from existing values or raw strikes."""
    fields: Dict[str, Any] = {}
    fields.update(_major_wall_gamma_fields(snapshot, prefer_positive=True))
    fields.update(_major_wall_gamma_fields(snapshot, prefer_positive=False))
    raw_strikes = snapshot.get("strikes")
    parsed_strikes = parse_gex_strikes(raw_strikes)

    # When a raw ladder is present it is authoritative. Recompute candidates
    # from volume gamma so provider-supplied/OI-derived compact fields cannot
    # leak into any symbol's pos_can*/neg_can* values.
    if isinstance(raw_strikes, list) and raw_strikes:
        for prefix, entries in (
            (
                "pos",
                summarize_wall_candidates(
                    snapshot.get("major_pos_vol"),
                    parsed_strikes,
                    prefer_positive=True,
                ),
            ),
            (
                "neg",
                summarize_wall_candidates(
                    snapshot.get("major_neg_vol"),
                    parsed_strikes,
                    prefer_positive=False,
                ),
            ),
        ):
            for idx in (1, 2):
                entry = entries[idx - 1] if idx - 1 < len(entries) else {}
                for field in ("strike", "value", "pct"):
                    fields[f"{prefix}_can{idx}_{field}"] = entry.get(field)
        return fields

    compact_keys = (
        ("pos", 1, "strike"),
        ("pos", 1, "value"),
        ("pos", 1, "pct"),
        ("pos", 2, "strike"),
        ("pos", 2, "value"),
        ("pos", 2, "pct"),
        ("neg", 1, "strike"),
        ("neg", 1, "value"),
        ("neg", 1, "pct"),
        ("neg", 2, "strike"),
        ("neg", 2, "value"),
        ("neg", 2, "pct"),
    )
    for side, idx, field in compact_keys:
        key = f"{side}_can{idx}_{field}"
        value = snapshot.get(key)
        if value is not None:
            fields[key] = _bounded_pct(value) if field == "pct" else value
    if len(fields) == len(compact_keys) + 2:
        return fields

    legacy_map = {
        "pos_can1_strike": ("call_wall_candidate1_strike",),
        "pos_can1_value": ("call_wall_candidate1_value", "call_wall_candidate1_gamma"),
        "pos_can1_pct": ("call_wall_candidate1_pct",),
        "pos_can2_strike": ("call_wall_candidate2_strike",),
        "pos_can2_value": ("call_wall_candidate2_value", "call_wall_candidate2_gamma"),
        "pos_can2_pct": ("call_wall_candidate2_pct",),
        "neg_can1_strike": ("put_wall_candidate1_strike",),
        "neg_can1_value": ("put_wall_candidate1_value", "put_wall_candidate1_gamma"),
        "neg_can1_pct": ("put_wall_candidate1_pct",),
        "neg_can2_strike": ("put_wall_candidate2_strike",),
        "neg_can2_value": ("put_wall_candidate2_value", "put_wall_candidate2_gamma"),
        "neg_can2_pct": ("put_wall_candidate2_pct",),
    }
    for compact_key, legacy_keys in legacy_map.items():
        value = fields.get(compact_key)
        if value is None:
            for legacy_key in legacy_keys:
                value = snapshot.get(legacy_key)
                if value is not None:
                    break
        if value is not None:
            fields[compact_key] = value
    if len(fields) == len(compact_keys):
        return fields

    pos_entries = summarize_wall_candidates(
        snapshot.get("major_pos_vol"), parsed_strikes, prefer_positive=True
    )
    neg_entries = summarize_wall_candidates(
        snapshot.get("major_neg_vol"), parsed_strikes, prefer_positive=False
    )
    for prefix, entries in (("pos", pos_entries), ("neg", neg_entries)):
        for idx in (1, 2):
            entry = entries[idx - 1] if idx - 1 < len(entries) else {}
            for field in ("strike", "value", "pct"):
                key = f"{prefix}_can{idx}_{field}"
                if key in fields and fields[key] is not None:
                    continue
                fields[key] = entry.get(field)
    return fields


def build_wall_ladder_from_compact(snapshot: Dict[str, Any], side: str) -> Dict[str, Any]:
    """Build the gex-monitor ladder payload from compact wall fields."""
    normalized_side = "pos" if side.lower() in {"call", "pos", "positive"} else "neg"
    # This is a presentation adapter. Candidate values must already have been
    # calculated on snapshot creation; downstream consumers only read them.
    compact_fields = {
        f"{normalized_side}_can{idx}_{field}": snapshot.get(
            f"{normalized_side}_can{idx}_{field}"
        )
        for idx in (1, 2)
        for field in ("strike", "value", "pct")
    }
    major_key = "major_pos_vol" if normalized_side == "pos" else "major_neg_vol"
    ladder = {"major": snapshot.get(major_key), "next": []}
    for idx in (1, 2):
        strike = compact_fields.get(
            f"{normalized_side}_can{idx}_strike", snapshot.get(f"{normalized_side}_can{idx}_strike")
        )
        value = compact_fields.get(
            f"{normalized_side}_can{idx}_value", snapshot.get(f"{normalized_side}_can{idx}_value")
        )
        pct = compact_fields.get(
            f"{normalized_side}_can{idx}_pct", snapshot.get(f"{normalized_side}_can{idx}_pct")
        )
        if strike is None and value is None and pct is None:
            continue
        ladder["next"].append({"strike": strike, "value": value, "pct": pct})
    return ladder
