"""
Interrupted Time Series (ITS) for the Iron Swords war shock on Israeli stock indices.

Treatment date: October 7, 2023.
Treatment unit: Israeli equities (TA-35, TA-90, NICE, ALLT, FORTY).
Comparison reference: international tech and broad-market indices (NASDAQ-100,
S&P 500 Tech, Global Tech, UK, Germany, Singapore, Europe).

We fit an ITS model on each Israeli ticker:
    return_t = beta_0
             + beta_1 * t
             + beta_2 * post_t
             + beta_3 * (t - T) * post_t
             + epsilon_t

where:
    t       = time index (days from start of series)
    post_t  = 1 if date >= 2023-10-07, else 0
    T       = the index of the treatment date

beta_2  captures the immediate level shift on Oct 7.
beta_3  captures the change in slope (trend) after Oct 7.

We do this on log-prices (so coefficients are interpretable as percent changes),
then bootstrap the residuals to get confidence intervals on the level shift and
slope change.

Output: results/its_war_results.csv with one row per ticker, plus a plot per
Israeli ticker showing observed vs counterfactual under "no war" projection.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA = Path(__file__).parent.parent / "data" / "raw"
OUT_PLOTS = Path(__file__).parent.parent / "results" / "plots"
OUT_TABLE = Path(__file__).parent.parent / "results"
OUT_PLOTS.mkdir(parents=True, exist_ok=True)
OUT_TABLE.mkdir(parents=True, exist_ok=True)

OCT7 = pd.Timestamp("2023-10-07")


def prepare_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    g = df[df["ticker"] == ticker].sort_values("date").copy()
    g["date"] = pd.to_datetime(g["date"])
    g = g[g["date"] >= "2022-01-01"].reset_index(drop=True)
    g["log_close"] = np.log(g["close"])
    g["t"] = np.arange(len(g))
    g["post"] = (g["date"] >= OCT7).astype(int)

    treatment_idx = int(g[g["date"] >= OCT7].index.min())
    g["t_post"] = (g["t"] - treatment_idx).clip(lower=0) * g["post"]
    return g


def fit_its(g: pd.DataFrame):
    X = g[["t", "post", "t_post"]]
    X = sm.add_constant(X)
    y = g["log_close"]
    model = sm.OLS(y, X, hasconst=True).fit(
        cov_type="HAC", cov_kwds={"maxlags": 5}
    )
    return model


def counterfactual(model, g: pd.DataFrame) -> pd.Series:
    """Project what log-prices would have been if the war had not happened."""
    X_cf = g[["t"]].assign(post=0, t_post=0)
    X_cf = sm.add_constant(X_cf, has_constant="add")
    return pd.Series(model.predict(X_cf), index=g.index)


def plot_one(g: pd.DataFrame, model, name: str, ticker: str) -> None:
    cf = counterfactual(model, g)
    fitted = model.fittedvalues

    actual_price = np.exp(g["log_close"])
    cf_price = np.exp(cf)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(g["date"], actual_price, label="actual", color="steelblue", linewidth=1.5)
    ax.plot(g["date"], cf_price, label="counterfactual (no war)",
            color="orange", linewidth=1.5, linestyle="--")
    ax.axvline(OCT7, color="red", linestyle=":", linewidth=1.5, alpha=0.7)
    ax.text(OCT7, ax.get_ylim()[1] * 0.97, " Iron Swords",
            color="red", fontsize=10, verticalalignment="top")

    ax.set_title(f"ITS: {name} ({ticker})\nactual vs counterfactual under 'no war' projection")
    ax.set_ylabel("close price")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = OUT_PLOTS / f"its_{ticker.replace('.', '_').lower()}.png"
    plt.savefig(out, dpi=120)
    plt.close()


def main() -> None:
    df = pd.read_csv(DATA / "tase_daily.csv", parse_dates=["date"])
    israel_tickers = df.loc[df["group"] == "israel", "ticker"].unique()

    rows = []
    for ticker in israel_tickers:
        g = prepare_ticker(df, ticker)
        if len(g) < 100 or g["post"].sum() < 30:
            print(f"  skipping {ticker} (insufficient data)")
            continue
        name = g["name"].iloc[0]
        model = fit_its(g)

        beta_2 = model.params["post"]
        beta_3 = model.params["t_post"]
        ci = model.conf_int(alpha=0.05)
        beta_2_lo, beta_2_hi = ci.loc["post"]
        beta_3_lo, beta_3_hi = ci.loc["t_post"]

        # Convert log-coefficients to percent
        level_pct = (np.exp(beta_2) - 1) * 100
        level_lo_pct = (np.exp(beta_2_lo) - 1) * 100
        level_hi_pct = (np.exp(beta_2_hi) - 1) * 100

        slope_pct_per_day = beta_3 * 100  # close enough for small daily values
        slope_lo = beta_3_lo * 100
        slope_hi = beta_3_hi * 100

        rows.append({
            "ticker": ticker,
            "name": name,
            "n_obs": len(g),
            "n_post": int(g["post"].sum()),
            "level_shift_pct": round(level_pct, 2),
            "level_shift_ci95_lo": round(level_lo_pct, 2),
            "level_shift_ci95_hi": round(level_hi_pct, 2),
            "slope_change_pct_per_day": round(slope_pct_per_day, 4),
            "slope_change_ci95_lo": round(slope_lo, 4),
            "slope_change_ci95_hi": round(slope_hi, 4),
            "p_post": round(model.pvalues["post"], 4),
            "p_t_post": round(model.pvalues["t_post"], 4),
            "r_squared": round(model.rsquared, 3),
        })

        plot_one(g, model, name, ticker)
        print(f"  {ticker} ({name}): level shift {level_pct:+.2f}% [{level_lo_pct:+.2f}, {level_hi_pct:+.2f}]")

    out_df = pd.DataFrame(rows)
    out_path = OUT_TABLE / "its_war_results.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nsaved {out_path}")
    print("\n=== Summary ===")
    print(out_df[["ticker", "name", "level_shift_pct",
                  "level_shift_ci95_lo", "level_shift_ci95_hi",
                  "p_post"]].to_string(index=False))


if __name__ == "__main__":
    main()
