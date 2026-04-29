import sys
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "trading" / "tradeslist_stop_sweep.py"
)
SPEC = spec_from_file_location("tradeslist_stop_sweep", MODULE_PATH)
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TradesListStopSweepTests(unittest.TestCase):
    def test_simulated_trade_pnl_includes_commission_on_stop(self) -> None:
        trade = MODULE.Trade(
            entry_date="2026-03-01",
            symbol="MNQH26_FUT_CME",
            trade_type="Long",
            pnl_dollar=24.16,
            commission_dollar=0.84,
            mae_dollar=-6.00,
            quantity=2,
            tick_value=0.50,
            duration="00:00:01",
        )

        pnl, stopped = MODULE.simulated_trade_pnl(
            trade,
            stop_ticks=5,
            slippage_ticks=0,
        )

        self.assertTrue(stopped)
        self.assertEqual(pnl, -5.84)

    def test_load_trades_uses_requested_mae_and_quantity_columns(self) -> None:
        tmp_dir = Path(__file__).resolve().parent / ".tmp"
        tmp_dir.mkdir(exist_ok=True)
        sample = tmp_dir / "TradesList-sample.txt"
        self.addCleanup(sample.unlink, missing_ok=True)

        sample.write_text(
            "\t".join(
                [
                    "Symbol",
                    "Trade Type",
                    "Entry DateTime",
                    "Profit/Loss (C)",
                    "Commission (C)",
                    "Max Open Loss (C)",
                    "FlatToFlat Max Open Loss (C)",
                    "Trade Quantity",
                    "Max Open Quantity",
                    "Duration",
                ]
            )
            + "\n"
            + "\t".join(
                [
                    "MNQH26_FUT_CME",
                    "Long",
                    "2026-03-01  18:16:11.148 BP",
                    "24.16",
                    "0.84",
                    "-2.50",
                    "-10.00",
                    "5",
                    "3",
                    "00:00:01",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        trades = MODULE.load_trades(
            sample,
            tick_size=0.25,
            point_value=None,
            tick_value=None,
            mae_column="flat-to-flat-max-open-loss",
            quantity_column="max-open-quantity",
        )

        self.assertEqual(len(trades), 1)
        trade = trades[0]
        self.assertEqual(trade.mae_dollar, -10.0)
        self.assertEqual(trade.quantity, 3)
        self.assertEqual(trade.tick_value, 0.5)


if __name__ == "__main__":
    unittest.main()
