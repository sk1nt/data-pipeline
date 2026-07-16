from src.services.gex_wall_utils import build_compact_wall_fields, summarize_wall_candidates


def test_wall_percentages_use_strongest_same_side_gamma() -> None:
    entries = summarize_wall_candidates(
        100,
        [(100, 5.0), (101, 10.0), (102, 7.5)],
        prefer_positive=True,
    )

    assert [entry["pct"] for entry in entries] == [100.0, 75.0]


def test_supplied_wall_percentages_are_bounded() -> None:
    fields = build_compact_wall_fields(
        {
            "pos_can1_pct": 125.0,
            "pos_can2_pct": -5.0,
            "neg_can1_pct": "150",
            "neg_can2_pct": "bad",
        }
    )

    assert fields["pos_can1_pct"] == 100.0
    assert fields["pos_can2_pct"] == 0.0
    assert fields["neg_can1_pct"] == 100.0
    assert fields["neg_can2_pct"] is None


def test_major_wall_strike_fields_are_populated_from_strikes() -> None:
    fields = build_compact_wall_fields(
        {
            "major_pos_vol": 29700.36,
            "major_neg_vol": 29680.36,
            "strikes": [
                [29700.36, 97995.36, -1911.78],
                [29710.36, 20578.96, -78.86],
                [29680.36, -5044.25, 60.69],
            ],
        }
    )

    assert fields["major_pos_vol_gamma"] == 97995.36
    assert fields["major_neg_vol_gamma"] == -5044.25
