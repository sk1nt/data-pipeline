#!/usr/bin/env python3
"""scripts/backtest_stop_system.py — evaluate the sweep-based stop system on
historical Sierra Chart TradesList data.

Simulates placing a stop at various tick distances and reports net P&L,
win rate, and per-trade statistics compared to the baseline (no stop).

Usage
─────
    python scripts/backtest_stop_system.py \\
        --trades trading/TradesList.txt \\
        --stops 5,10,15,20,25,30 \\
        --slippage 1 \\
        --tick-value 0.50 \\
        --mae-column flat-to-flat-max-open-loss \\
        --quantity-column max-open-quantity

    # Quick check with defaults:
    python scripts/backtest_stop_system.py --trades trading/TradesList.txt

Output
──────
    Console table comparing net P&L, win rate, stopped %, and average trade
    across each stop level.  Pass --csv to also write a results CSV.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Allow running from repo root without installing
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trading.tradeslist_stop_sweep import load_trades, simulated_trade_pnl, Trade  # noqa: E402


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def _run_stop_level(
    trades: list[Trade],
    stop_ticks: float,
    slippage_ticks: float,
) -> dict:
    total_pnl   = 0.0
    wins        = 0
    stopped     = 0
    avg_pnl_sum = 0.0

    for trade in trades:
        pnl, was_stopped = simulated_trade_pnl(trade, stop_ticks, slippage_ticks)
        total_pnl   += pnl
        avg_pnl_sum += pnl
        if pnl > 0:
            wins += 1
        if was_stopped:
            stopped += 1

    n = len(trades)
    return {
        "stop_ticks":   stop_ticks,
        "n_trades":     n,
        "net_pnl":      round(total_pnl,   2),
        "win_rate":     round(wins / n * 100,    1) if n else 0.0,
        "stop_rate":    round(stopped / n * 100, 1) if n else 0.0,
        "avg_trade":    round(avg_pnl_sum / n,   2) if n else 0.0,
        "wins":         wins,
        "stopped":      stopped,
    }


def run_backtest(
    trades: list[Trade],
    stop_levels: list[float],
    slippage_ticks: float = 0.0,
) -> list[dict]:
    """Return a list of result dicts — one per stop level plus baseline."""
    results = []
    # Baseline: no stop
    baseline = _run_stop_level(trades, stop_ticks=0, slippage_ticks=0)
    baseline["stop_ticks"] = 0
    baseline["label"] = "no stop"
    results.append(baseline)

    for ticks in sorted(stop_levels):
        r = _run_stop_level(trades, ticks, slippage_ticks)
        r["label"] = f"{ticks:g}t"
        results.append(r)

    return results


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _print_table(results: list[dict], baseline_pnl: float) -> None:
    header = (
        f"{'Stop':>8}  {'Net P&L':>10}  {'vs Base':>9}  "
        f"{'Win%':>6}  {'Stop%':>6}  {'Avg $':>8}  {'Trades':>7}"
    )
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        delta = r["net_pnl"] - baseline_pnl
        delta_str = f"{delta:+.2f}" if r["stop_ticks"] > 0 else "  base"
        print(
            f"{r['label']:>8}  {r['net_pnl']:>10.2f}  {delta_str:>9}  "
            f"{r['win_rate']:>5.1f}%  {r['stop_rate']:>5.1f}%  "
            f"{r['avg_trade']:>8.2f}  {r['n_trades']:>7}"
        )
    print(sep)


def _write_csv(results: list[dict], out_path: Path) -> None:
    fields = ["label", "stop_ticks", "n_trades", "net_pnl", "win_rate",
              "stop_rate", "avg_trade", "wins", "stopped"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV written → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backtest stop-loss levels on SC TradesList export",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--trades", required=True, metavar="FILE",
        help="Path to Sierra Chart TradesList .txt export",
    )
    p.add_argument(
        "--stops", default="5,10,15,20,25,30", metavar="TICKS",
        help="Comma-separated stop levels in ticks (default: 5,10,15,20,25,30)",
    )
    p.add_argument(
        "--slippage", type=float, default=0.0, metavar="TICKS",
        help="Slippage added to each stop trigger (default: 0)",
    )
    p.add_argument(
        "--tick-value", type=float, default=None, metavar="$",
        help="Dollar value per tick per contract (e.g. 0.50 for MNQ). "
             "Auto-inferred from symbol if omitted.",
    )
    p.add_argument(
        "--tick-size", type=float, default=None, metavar="POINTS",
        help="Price per tick (e.g. 0.25). Used with --point-value.",
    )
    p.add_argument(
        "--point-value", type=float, default=None, metavar="$",
        help="Dollar value per 1.0 price point (e.g. 2.0 for MNQ).",
    )
    p.add_argument(
        "--mae-column",
        default="max-open-loss",
        choices=["max-open-loss", "flat-to-flat-max-open-loss"],
        help="Which MAE column to use for stop simulation (default: max-open-loss)",
    )
    p.add_argument(
        "--quantity-column",
        default="trade-quantity",
        choices=["trade-quantity", "max-open-quantity"],
        help="Which quantity column to use (default: trade-quantity)",
    )
    p.add_argument(
        "--csv", metavar="FILE",
        help="Also write results to this CSV path",
    )
    p.add_argument(
        "--filter-symbol", metavar="SYM",
        help="Only include trades whose symbol contains SYM (case-insensitive)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    trades_path = Path(args.trades)
    if not trades_path.exists():
        print(f"Error: {trades_path} not found", file=sys.stderr)
        sys.exit(1)

    try:
        stop_levels = [float(x.strip()) for x in args.stops.split(",") if x.strip()]
    except ValueError:
        print("Error: --stops must be comma-separated numbers", file=sys.stderr)
        sys.exit(1)

    trades = load_trades(
        trades_path,
        tick_size=args.tick_size,
        point_value=args.point_value,
        tick_value=args.tick_value,
        mae_column=args.mae_column,
        quantity_column=args.quantity_column,
    )

    if not trades:
        print("No trades loaded — check the file format.", file=sys.stderr)
        sys.exit(1)

    if args.filter_symbol:
        sym_filter = args.filter_symbol.upper()
        trades = [t for t in trades if sym_filter in t.symbol.upper()]
        if not trades:
            print(f"No trades left after filtering for '{args.filter_symbol}'", file=sys.stderr)
            sys.exit(1)

    print(f"\nLoaded {len(trades)} trades from {trades_path.name}")
    if args.filter_symbol:
        print(f"Filtered to symbol: {args.filter_symbol}")
    print(f"Slippage: {args.slippage} ticks")
    print(f"MAE column: {args.mae_column}")
    print(f"Quantity column: {args.quantity_column}")
    print()

    results = run_backtest(trades, stop_levels, slippage_ticks=args.slippage)
    baseline_pnl = results[0]["net_pnl"]
    _print_table(results, baseline_pnl)

    if args.csv:
        _write_csv(results, Path(args.csv))


if __name__ == "__main__":
    main()
