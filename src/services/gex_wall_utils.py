"""Helpers for deriving compact GEX wall candidate payloads."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


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
                if gamma == 0.0 and len(entry) >= 3:
                    gamma = float(entry[2])
                strikes.append((strike, gamma))
            except (TypeError, ValueError):
                continue
        elif isinstance(entry, dict):
            try:
                strike = float(entry.get("strike"))
                gamma = float(entry.get("gamma"))
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

    entries: List[Dict[str, Any]] = []
    for strike, gamma in filtered:
        if abs(strike - major[0]) <= 0.51:
            continue
        entries.append(
            {
                "strike": strike,
                "value": gamma,
                "pct": abs(gamma) / abs(major[1]) * 100 if major[1] else None,
            }
        )
        if len(entries) >= 2:
            break
    return entries


def build_compact_wall_fields(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Populate compact wall fields from existing values or raw strikes."""
    fields: Dict[str, Any] = {}
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
            fields[key] = value
    if len(fields) == len(compact_keys):
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

    strikes = parse_gex_strikes(snapshot.get("strikes"))
    pos_entries = summarize_wall_candidates(
        snapshot.get("major_pos_vol"), strikes, prefer_positive=True
    )
    neg_entries = summarize_wall_candidates(
        snapshot.get("major_neg_vol"), strikes, prefer_positive=False
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
    compact_fields = build_compact_wall_fields(snapshot)
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
