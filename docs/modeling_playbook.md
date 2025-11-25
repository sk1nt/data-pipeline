How to Incorporate GEX Effectively
==================================

Scope
-----
- Data: MNQ/NQ intraday (1s–500-contract bars), dealer positioning for NDX (net GEX, walls, zero‑gamma flip), basic tech features.
- Goal: directional / expected-move models that adapt to GEX regime, cost-aware, ready for RTH live use.

Data Framing & Alignment
------------------------
- Bars: prefer volume or dollar bars on heavy‑tick days; time bars are fine when flow is light. State bar type & size in each run.
- GEX sync: forward-fill to 1–5s cadence; max staleness 30–60s; drop samples when stale to avoid regime drift.
- Time features: Δt since last bar, minutes from RTH open, RTH flag.
- Price/GEX geometry: distance to strongest |GEX| strike, distance to zero‑gamma, distance to nearest wall above/below.

Feature Set (22-dim current GEX subset)
open, high, low, close, volume, gex_zero, net_gex, major_pos_vol, major_neg_vol, sum_gex_vol, delta_risk_reversal, candidate_pos_strike, candidate_neg_strike, candidate_pos_vol, candidate_neg_vol, max_priors_current, max_priors_1m, max_priors_5m, williams_r, rsi, bb_upper, bb_lower.

Labels & Cost Awareness
-----------------------
- Binary: future return > commission+slippage break-even (currently 1% = 0.01) over horizon h (default 1s); use cost-aware labels.
- Thresholding: sweep 0.50–0.60; calibrate (Platt/Isotonic) before choosing a cut.
- Evaluation: walk-forward with purge; RTH slices (e.g., 09:45–11:30 ET); include TP/SL grids in PnL sims.

Model Family Cheat Sheet
------------------------
1) Transformers (TFT/Informer/linear attention)
   - Use time-delta positional encodings; chunked or performer-style attention for seq >512.
   - Treat GEX curve as tokens or channels concatenated to bar tokens.
2) LSTM/GRU + Attention
   - Add a small “regime head” gating on net GEX sign/magnitude; avoid bidirectional at inference to prevent leakage.
3) TCN / Dilated CNN
   - Causal padding; great when LOB+GEX are rasterized into price × time grids.
4) Deep RL (PPO/SAC)
   - State = bar/LOB slice + GEX regime; reward = cost-shaped PnL; clip actions; session-sized replay to limit drift.
5) GBDT (LightGBM/XGBoost/CatBoost)
   - Fast, robust baseline; enforce priors via monotone constraints (distance-from-wall ↓ → win-prob ↓); always calibrate + threshold.

Recommended Hyperparams (ranges)
--------------------------------
- Transformers: d_model 64–256, heads 4–8, layers 2–4, seq len 256–1024, dropout 0.1–0.2, linear/chunked attention if seq >512.
- LSTM/GRU: hidden 64–256, layers 1–3, dropout 0–0.3; attention head 32–64.
- TCN: kernel 3–5, layers 6–10 with doubling dilation, channels 32–128.
- PPO baseline: lr 1e-4–3e-4, clip 0.1–0.2, γ 0.95–0.99, λ 0.9–0.95; reward includes commission+slippage.
- GBDT: trees 300–800, depth 4–8, lr 0.03–0.1, subsample 0.6–0.9; calibrate, then threshold 0.54–0.58 for RTH.

Execution Tips
--------------
- Latency: keep inference < a few ms; bar data to reduce variance.
- Overfitting: walk-forward, no look-ahead; always bake costs into labels or loss.
- Regime routing: train a tiny classifier on net GEX → choose mean-reversion model in +GEX and momentum model in –GEX.

Next Steps
----------
- Train/update regime classifier; export router.
- Backtest with RTH window + TP/SL grids.
- Add Make/CLI targets for cost-aware preprocessing and RTH backtests.
