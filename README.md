Technical & Econometric Architecture

Feature Engineering:

Order Book Imbalance (OBI): Measures supply/demand volume asymmetry at top-of-book.

Micro-Price Deviation (MPD): Captures the deviation of the volume-weighted micro-price from the mid-price.

Diagnostics:

Augmented Dickey-Fuller (ADF) Tests: Rejects unit roots (p<0.001) to ensure feature stationarity before calibration.

Newey-West HAC Standard Errors: Corrects t-statistics for serial correlation induced by overlapping forward return windows.

Model Calibration:

Exponentially Weighted Least Squares (EWLS): Fits return-prediction parameters (β) with exponential decay to prioritize recent order flow regimes over stale historical data.

Strategy Backtest & Key Microstructure Flaws

Out-of-sample backtest results yielded weak performance metrics (Negative PnL and Sharpe ratio), driven by several practical market microstructure challenges:

Prediction Horizon vs. Friction Trade-off:

Short Horizons: Frequent trading quickly erased alpha due to bid-ask spread crossing and fixed transaction fees.

Long Horizons: Forecasted alpha decayed before realization; mid-horizon position flips forced early exits and signal invalidation before full return convergence.

Static Hyperparameters: The EWLS half-life parameter was fixed manually rather than dynamically adjusted to real-time volatility regimes.

Underutilized L2 Depth: Execution quantities were capped at 1 unit, avoiding L2 book depth consumption and market impact modeling.

Future Roadmap & Institutional Solutions

To address these empirical limitations, planned updates to the framework include:

Overlapping Sub-Portfolios: Implementing staggered sub-accounts to allocate capital across different target execution horizons, preventing mid-horizon signal overrides.

Adaptive Half-Life Tuning: Dynamically calibrating EWLS decay speed using volatility and order flow intensity metrics.

L2 Depth Impact Modeling: Expanding the execution engine to model walking the book across Level-2 queues and incorporating dynamic queue position estimation.
