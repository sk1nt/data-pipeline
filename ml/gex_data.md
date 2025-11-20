# GEX Data Fields Documentation

## Overview
This document details the Gamma Exposure (GEX) data fields used in the LSTM trading model. These fields capture market microstructure signals from options gamma positioning, providing insights into institutional hedging activity and market maker positioning.

**Data Source**: GEXBot API providing real-time options gamma calculations for NQ futures.  
**Storage**: See `GEX_HISTORY_PIPELINE.md` and `DATA_STRUCTURE_AND_GOVERNANCE.md` for complete pipeline details.  
**Usage**: NQ GEX data aligned with MNQ tick data for high-frequency trading signals.

## Data Pipeline Context

### Source Data Structure
- **Raw Format**: JSON files from GEXBot API (e.g., `2025-11-14_NQ_NDX_classic_gex_zero.json`)
- **Staging Location**: `data/source/gexbot/<ticker>/<endpoint>/`
- **Canonical Storage**: `data/parquet/gexbot/<ticker>/<endpoint>/<YYYYMMDD>.strikes.parquet`
- **Database Tables**:
  - `gex_snapshots`: Aggregate metrics per timestamp
  - `gex_strikes`: Flattened strike-level data with `max_priors` JSON field

### Processing Flow
1. **Download**: JSON snapshots from GEXBot API
2. **Staging**: Load into DuckDB `gex_snapshots` and `gex_strikes` tables
3. **Export**: Canonical Parquet files with ZSTD compression
4. **Alignment**: Forward-fill GEX data to match tick timestamps during preprocessing

## Core GEX Fields

### zero_gamma
Zero gamma is the price at which the market makers gamma exposure is zero
**Definition**: The gamma exposure at the current spot price (zero-strike gamma).
**Units**: Gamma per dollar of underlying price movement.
**Range**: Typically 25,000-26,000 for NQ/MNQ.
**Usage**: Primary measure of market maker hedging pressure. High values indicate strong gamma hedging that can create directional momentum.

### spot_price
**Definition**: The current underlying asset price (NQ futures).
**Units**: Price in cents (e.g., 2,550,000 = $25,500).
**Usage**: Reference price for all gamma calculations. Used to align GEX data with tick data timestamps.

### net_gex
**Definition**: Net gamma exposure across all strikes (positive gamma minus negative gamma).
**Units**: Gamma units.
**Range**: Typically -25,000 to 0 (net negative due to market maker positioning).
**Usage**: Overall market direction bias. Negative values indicate market makers are net short gamma, creating upward pressure on volatility expansion.

## Gamma Flow Fields

### major_pos_vol
**Definition**: Total gamma from positive gamma positions (calls and puts with positive gamma).
**Units**: Gamma units.
**Range**: Typically 25,000-26,000.
**Usage**: Measures bullish positioning strength. Higher values indicate more call buying or put selling activity.

### major_neg_vol
**Definition**: Total gamma from negative gamma positions (calls and puts with negative gamma).
**Units**: Gamma units.
**Range**: Typically 25,000-26,000.
**Usage**: Measures bearish positioning strength. Higher values indicate more put buying or call selling activity.

### sum_gex_vol
**Definition**: Net gamma flow (major_pos_vol - major_neg_vol).
**Units**: Gamma units.
**Range**: -42,000 to +21,000.
**Usage**: Directional gamma imbalance. Positive values show net bullish flows, negative values show net bearish flows.

## Risk Reversal Fields

### delta_risk_reversal
**Definition**: The difference between call and put delta hedging costs.
**Units**: Delta units (typically 0.00-0.30).
**Range**: -0.07 to +0.27.
**Usage**: Measures skew in hedging activity. Positive values indicate more call delta hedging (bullish), negative values indicate more put delta hedging (bearish).

## Max Priors Fields (Institutional Positioning)

### Data Structure
The `max_priors` fields are extracted from the `priors` JSON object in the `gex_strikes` table, which contains per-strike gamma change data. During preprocessing, we identify the strike with the maximum absolute gamma change for each timeframe.

### max_priors_current
**Definition**: Strike with the largest gamma change in the current second.
**Units**: Gamma change per second at that strike.
**Range**: -7,000 to +11,000.
**JSON Source**: `priors` object in `gex_strikes` table, current second delta.
**Usage**: Real-time institutional positioning. Large positive values indicate aggressive buying at that strike, large negative values indicate aggressive selling.

### max_priors_1m
**Definition**: Strike with the largest gamma change over the past minute.
**Units**: Gamma change per minute at that strike.
**Range**: -8,000 to +7,000.
**JSON Source**: `priors` object aggregated over 60 seconds.
**Usage**: Short-term positioning trends. Sustained large values indicate conviction in directional positioning.

### max_priors_5m
**Definition**: Strike with the largest gamma change over the past 5 minutes.
**Units**: Gamma change per 5 minutes at that strike.
**Range**: -15,000 to +5,000.
**JSON Source**: `priors` object aggregated over 300 seconds.
**Usage**: Medium-term positioning trends. Persistent large values reveal significant institutional accumulation or distribution.

## LSTM Model Integration

### Feature Engineering Pipeline
- **Input Data**: MNQ tick data (1-second OHLCV) aligned with NQ GEX snapshots
- **Sequence Length**: 60 timesteps (1 minute of market data)
- **Total Features**: 21 (5 OHLCV + 9 GEX + 7 technical indicators)
- **Target**: Binary classification (price direction in next second: up=1, down=0)

### GEX Feature Processing
- **Scaling**: StandardScaler applied to all features for neural network compatibility
- **Missing Data**: Forward-filled from last valid GEX snapshot
- **Market Hours**: Filtered to 9:32am-4:00pm ET trading session
- **Timezone**: UTC GEX timestamps converted to US/Eastern for alignment

### Model Architecture
- **Network**: PyTorch LSTM with 256 hidden units, 2 layers
- **Input Shape**: (batch_size, 60 timesteps, 21 features)
- **Output**: Sigmoid activation for binary price prediction
- **Training**: Supervised learning on historical sequences

### Feature Importance Insights
Based on correlation analysis and domain knowledge:
- **zero_gamma**: Primary hedging pressure signal
- **net_gex**: Overall market directional bias
- **max_priors_current**: Real-time institutional activity
- **sum_gex_vol**: Net gamma flow direction
- **delta_risk_reversal**: Hedging skew indicator

## Trading Applications

### Signal Generation
- **High zero_gamma + negative net_gex**: Strong upward momentum potential
- **Large positive max_priors_current**: Institutional buying pressure
- **Persistent negative max_priors_5m**: Institutional distribution signal
- **delta_risk_reversal > 0.15**: Bullish hedging skew

### Risk Management
- **Extreme max_priors values**: High conviction signals (potential for large moves)
- **zero_gamma spikes**: Volatility expansion events
- **net_gex magnitude**: Market maker positioning stress

## Data Quality & Validation

### Coverage Metrics
- **Market Hours Coverage**: 100% during 9:32am-4:00pm ET
- **Data Density**: ~1.5M valid records per trading day
- **Missing Values**: Forward-filled from previous snapshots
- **Outlier Handling**: Values within expected market microstructure ranges

### Statistical Validation
- **Realistic Ranges**: All fields exhibit expected gamma exposure distributions
- **Correlation Analysis**: Small but meaningful correlations with price returns (-0.0015 to +0.0013)
- **Stationarity**: Time-series properties suitable for LSTM modeling
- **Cross-validation**: Features show consistent behavior across different market conditions

### Quality Checks
- **zero_gamma**: 25,000-26,000 range (typical for NQ/MNQ)
- **net_gex**: Negative bias (-3,000 to 0) indicating market maker positioning
- **max_priors**: Large magnitude values (±1,000 to ±15,000) for institutional activity
- **delta_risk_reversal**: Tight range (0.00-0.30) for hedging skew

## References
- **Pipeline Details**: See `GEX_HISTORY_PIPELINE.md` for complete data flow
- **Data Governance**: See `DATA_STRUCTURE_AND_GOVERNANCE.md` for storage rules
- **Model Code**: See `ml/backtest_model.py` for implementation details
- **Data Source**: GEXBot API (https://hist.gex.bot/)</content>
<parameter name="filePath">/home/rwest/projects/data-pipeline/docs/gex_data.md
