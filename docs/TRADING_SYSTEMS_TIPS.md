Key Practical Tips for Live Trading Systems
=========================================

This short guide collects practical advice used in production trading systems. These are high-level operational lessons distilled from live experience.

- Avoid raw tick timestamps: use volume/time/dollar bars
  - Convert tick-level data to aggregated bars (volume/dollar bars or fixed-time bars) to reduce non-stationarity and heavy tails.
  - Aggregation improves modeling stability and can help reduce latency sensitivity on microsecond signals.

- Beware severe overfitting
  - Use walk-forward (rolling) training and test partitions with strict no-lookahead restrictions.
  - Ensure a realistic transaction-cost model: spread + commission + slippage (per /NQ, 0.25–0.50 tick is a typical latency/slippage assumption).
  - Track exposures and value-at-risk across rollouts, and use model ensembles to reduce variance.

- Latency is an edge killer
  - Even for 5–15 minute trades, being 50–200ms early/late can remove the edge for GEX-driven strategies.
  - Log and simulate realistic latencies (market data, signal, execution) in backtests.

- Start simple
  - Use robust, low-variance models like LightGBM on engineered features before moving to more complex DL models.
  - LightGBM often wins in live setups due to easier tuning and generalization.

Practical checks to add to codebase
- Use time-aggregated dataset generation in `ml/extract.py`/`ml/preprocess.py` as an option (volume or dollar bars) in addition to 1s bars.
- Include a `--commission-per-side` or `--slippage-ms` CLI option on scripts like `pnl_backtest.py` and `train_*` to make transaction-cost assumptions explicit and consistent.
- Add unit/regression tests that assert any new dataset or model run does not create `mlruns` or artifacts in repo root (CI guard already included; consider an automated test that runs a small sample).

References
- `ml/preprocess.py`, `ml/pnl_backtest.py`, `ml/extract.py`
- Consider article: 'Volume Bars and Dollar Bars for Trading' for more on bar types.

If you'd like, I can implement the following next: add a CLI flag in `ml/extract.py`/`ml/preprocess.py` to produce volume- or dollar-bars, add an optional commission/slippage CLI param to backtests, and add a small test case ensuring no root-level `mlruns` are created. Which should I do first?
