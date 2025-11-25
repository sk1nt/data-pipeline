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
- Keep dataset generation scripts flexible enough to produce both time-based bars and volume/dollar bars; this logic now lives in the standalone modeling repo that replaced the legacy `ml/` directory here.
- Include knobs for commissions/slippage in any backtest CLI so the assumptions stay explicit and consistent with live trading.
- Add unit/regression tests in the modeling repo to ensure training outputs and runs are redirected to their own workspace (no more guarding against root-level `mlruns` in this project).

References
- Modeling scripts now live in the dedicated ML repository.
- Consider article: 'Volume Bars and Dollar Bars for Trading' for more on bar types.
