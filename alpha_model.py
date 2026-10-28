import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# Micro features

def compute_order_book_imbalance(bid_vol: pd.Series, ask_vol: pd.Series) -> pd.Series:
    total_vol = bid_vol + ask_vol
    total_vol = total_vol.replace(0, np.nan)
    return (bid_vol - ask_vol) / total_vol

def compute_micro_price_deviation(bid_price: pd.Series, ask_price: pd.Series, bid_vol: pd.Series, ask_vol: pd.Series) -> pd.Series:
    total_vol = bid_vol + ask_vol
    total_vol = total_vol.replace(0, np.nan)
    p_mid = (bid_price + ask_price) / 2.0
    p_micro = (bid_price * ask_vol + ask_price * bid_vol) / total_vol
    return p_micro - p_mid

def compute_forward_return(p_mid: pd.Series, horizon: int = 5) -> pd.Series:
    p_future = p_mid.shift(-horizon)
    return (p_future - p_mid) / p_mid

def compute_newey_west_t_stat(x: pd.Series, y: pd.Series, max_lags: int = 5) -> float:
    valid_data = pd.DataFrame({"x": x, "y": y}).dropna()
    X = sm.add_constant(valid_data["x"])
    Y = valid_data["y"]
    model = sm.OLS(Y, X).fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})
    return float(model.tvalues["x"])

def compute_ewls_beta(X: np.ndarray, y: np.ndarray, halflife: float) -> np.ndarray:
    N = len(y)
    y = y.reshape(-1, 1)
    lamda_decay = np.log(2) / halflife
    delta_t = (N - 1) - np.arange(N)
    weights = np.exp(-lamda_decay * delta_t)

    W_sqrt = np.sqrt(weights)[:, None]
    X_Weighted = X * W_sqrt
    y_Weighted = y * W_sqrt
    return np.linalg.solve(X_Weighted.T @ X_Weighted, X_Weighted.T @ y_Weighted)

def compute_adf_test(series: pd.Series) -> dict:
    clean_series = series.dropna()
   res = adfuller(clean_series, autolag="AIC")
    return {
        "adf_stat": float(res[0]),
        "p_value": float(res[1]),
        "is_stationary": bool(res[1] < 0.05)
    }

# Synthetic LOB generator

def generate_gbm_mid_price(
    s0: float = 100.0,
    mu: float = 0.0,
    sigma: float = 0.85,
    dt: float = 1.0 / 252,
    n_ticks: int = 1000,
    seed: int = 42
) -> np.ndarray:
    np.random.seed(seed)
    z = np.random.randn(n_ticks)
    drift = (mu - 0.5 * (sigma ** 2)) * dt
    diffusion = sigma * np.sqrt(dt) * z
    return s0 * np.exp(np.cumsum(drift + diffusion))

def generate_synthetic_lob(
    n_ticks: int = 1000,
    s0: float = 100.0,
    tick_size: float = 0.01,
    seed: int = 42
) -> pd.DataFrame:
    np.random.seed(seed)
    mid_prices = generate_gbm_mid_price(s0=s0, n_ticks=n_ticks, seed=seed)

    spread_ticks = np.random.choice([1, 2, 3], size=n_ticks, p=[0.7, 0.2, 0.1])
    half_spread = (spread_ticks * tick_size) / 2.0

    bid_p1 = np.round((mid_prices - half_spread) / tick_size) * tick_size
    ask_p1 = np.round((mid_prices + half_spread) / tick_size) * tick_size
    ask_p1 = np.maximum(ask_p1, bid_p1 + tick_size)

    bid_p2 = bid_p1 - tick_size
    ask_p2 = ask_p1 + tick_size

    bid_v1 = np.random.gamma(shape=3.0, scale=100.0, size=n_ticks).astype(int) + 10
    ask_v1 = np.random.gamma(shape=3.0, scale=100.0, size=n_ticks).astype(int) + 10
    bid_v2 = np.random.gamma(shape=4.0, scale=150.0, size=n_ticks).astype(int) + 20
    ask_v2 = np.random.gamma(shape=4.0, scale=150.0, size=n_ticks).astype(int) + 20

    return pd.DataFrame({
        "bid_p1": bid_p1, "ask_p1": ask_p1, "bid_v1": bid_v1, "ask_v1": ask_v1,
        "bid_p2": bid_p2, "ask_p2": ask_p2, "bid_v2": bid_v2, "ask_v2": ask_v2
    })

# Execution engine and metrics

def simulate_execution(
    lob_df: pd.DataFrame,
    y_hat: pd.Series,
    commission_per_share: float = 0.0001,
    inventory_limit: int = 1,
) -> pd.DataFrame:
    n = len(lob_df)
    position = np.zeros(n)
    cash = np.zeros(n)
    holdings = np.zeros(n)
    pnl = np.zeros(n)
    trades = np.zeros(n)

    current_pos = 0
    current_cash = 0.0

    for t in range(n):
        bid_p1 = lob_df["bid_p1"].iloc[t]
        ask_p1 = lob_df["ask_p1"].iloc[t]
        mid_p = (bid_p1 + ask_p1) / 2.0
        spread = ask_p1 - bid_p1

        half_spread = spread / 2.0
        theta_t = (half_spread + commission_per_share) / mid_p
        pred_ret = y_hat.iloc[t]

        if np.isnan(pred_ret):
            target_pos = 0
        elif pred_ret > theta_t:
            target_pos = inventory_limit
        elif pred_ret < -theta_t:
            target_pos = -inventory_limit
        else:
            target_pos = current_pos

        trade_units = target_pos - current_pos

        if trade_units != 0:
            exec_price = ask_p1 if trade_units > 0 else bid_p1
            transaction_cost = abs(trade_units) * commission_per_share
            cash_flow = -(trade_units * exec_price) - transaction_cost

            current_cash += cash_flow
            current_pos = target_pos
            trades[t] = trade_units

        position[t] = current_pos
        cash[t] = current_cash
        holdings[t] = current_pos * mid_p
        pnl[t] = cash[t] + holdings[t]

    return pd.DataFrame({
        "position": position, "cash": cash, "holdings": holdings,
        "pnl": pnl, "trades": trades
    }, index=lob_df.index)

def compute_quant_performance(execution_df: pd.DataFrame) -> dict:
    pnl = execution_df["pnl"]
    returns = pnl.diff().fillna(0)

    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = ((mean_ret / std_ret) * np.sqrt(252 * 23400)) if std_ret > 0 else 0.0

    peak = pnl.cummax()
    drawdown = pnl - peak
    max_drawdown = drawdown.min()
    total_trades = int(np.abs(execution_df["trades"]).sum())

    return {
        "Total Realized PnL ($)": round(float(pnl.iloc[-1]), 4),
        "Sharpe Ratio": round(float(sharpe), 2),
        "Max Drawdown ($)": round(float(max_drawdown), 4),
        "Total Executed Trades": total_trades,
    }

# Execution

if __name__ == "__main__":

    train_lob = generate_synthetic_lob(n_ticks=1000, seed=42)
    obi_train = compute_order_book_imbalance(train_lob["bid_v1"], train_lob["ask_v1"])
    mpd_train = compute_micro_price_deviation(
        train_lob["bid_p1"], train_lob["ask_p1"], train_lob["bid_v1"], train_lob["ask_v1"]
    )
    mid_train = (train_lob["bid_p1"] + train_lob["ask_p1"]) / 2.0
    fwd_ret_train = compute_forward_return(mid_train, horizon=5)

    train_df = pd.DataFrame({"obi": obi_train, "mpd": mpd_train, "fwd_ret": fwd_ret_train}).dropna()

    print("ADF Stationarity Test")
    print("OBI ADF:", compute_adf_test(train_df["obi"]))
    print("MPD ADF:", compute_adf_test(train_df["mpd"]))

    print("\nNewey-West HAC t-Stats")
    print("OBI t-stat:", compute_newey_west_t_stat(train_df["obi"], train_df["fwd_ret"]))
    print("MPD t-stat:", compute_newey_west_t_stat(train_df["mpd"], train_df["fwd_ret"]))

    X_raw = train_df[["obi", "mpd"]]
    X_matrix = sm.add_constant(X_raw).values
    y_vector = train_df["fwd_ret"].values

    calibrated_betas = compute_ewls_beta(X_matrix, y_vector, halflife=100.0)
    b0, b1, b2 = calibrated_betas[0, 0], calibrated_betas[1, 0], calibrated_betas[2, 0]

    print("\nEWLS Calibrated Betas")
    print(f"Intercept (b0): {b0:.8f}")
    print(f"OBI Beta  (b1): {b1:.8f}")
    print(f"MPD Beta  (b2): {b2:.8f}")

    # Actual test

    test_lob = generate_synthetic_lob(n_ticks=20000, seed=100)

    obi_test = compute_order_book_imbalance(test_lob["bid_v1"], test_lob["ask_v1"])
    mpd_test = compute_micro_price_deviation(
        test_lob["bid_p1"], test_lob["ask_p1"], test_lob["bid_v1"], test_lob["ask_v1"]
    )

    pred_returns = b0 + (b1 * obi_test) + (b2 * mpd_test)

    execution_df = simulate_execution(
        lob_df=test_lob,
        y_hat=pred_returns,
        commission_per_share=0.0001,
        inventory_limit=1,
    )

    metrics = compute_quant_performance(execution_df)

    print("\nSTRATEGY BACKTEST RESULTS")
    print(f"Final Total PnL:        ${metrics['Total Realized PnL ($)']:,.4f}")
    print(f"Total Executed Trades:  {metrics['Total Executed Trades']}")
    print(f"Annualized Sharpe Ratio:{metrics['Sharpe Ratio']:.2f}")
    print(f"Maximum Drawdown (MtM): ${metrics['Max Drawdown ($)']:,.4f}")

# --- Information Coefficient (IC) Evaluation ---
import scipy.stats as stats

train_df["fwd_ret"] = train_df["mid_price"].pct_change().shift(-1) if "mid_price" in train_df.columns else fwd_ret_train

# 2. Filter out NaNs, Infs, and zero price rows
clean_ic_df = train_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["fwd_ret", "obi", "mpd"])

# 3. Compute Pearson and Spearman ICs
pearson_obi, _ = stats.pearsonr(clean_ic_df["obi"], clean_ic_df["fwd_ret"])
pearson_mpd, _ = stats.pearsonr(clean_ic_df["mpd"], clean_ic_df["fwd_ret"])

spearman_obi, _ = stats.spearmanr(clean_ic_df["obi"], clean_ic_df["fwd_ret"])
spearman_mpd, _ = stats.spearmanr(clean_ic_df["mpd"], clean_ic_df["fwd_ret"])

print("\n--- Information Coefficients ---")
print(f"OBI Pearson IC:   {pearson_obi:.4f}  |  Spearman Rank IC: {spearman_obi:.4f}")
print(f"MPD Pearson IC:   {pearson_mpd:.4f}  |  Spearman Rank IC: {spearman_mpd:.4f}")
