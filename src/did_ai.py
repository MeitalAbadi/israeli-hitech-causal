"""
Difference-in-Differences for the GPT-4 release shock on tech vs non-tech equities.

Treatment date:    GPT-4 release, 2023-03-14.
Treatment group:   tech and software equities (NICE, XLK, IXN, QQQ).
Control group:     broad-market non-tech (TA-35, EWU, EWG, EWS, VGK).
Window:            September 2022 to September 2023 (6 months on each side).
                   We deliberately stop pre-Oct-7 to avoid the war shock contaminating
                   the post-period.

Model:
    log_price = beta_0
              + beta_1 * tech
              + beta_2 * post
              + beta_3 * (tech * post)
              + epsilon

beta_3 is the DiD effect: additional change in tech relative to non-tech
after GPT-4 was released.

Output: results/did_ai_results.csv with the DiD coefficient + CI + p-value,
and a plot showing the parallel-trends story.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA = Path(__file__).parent.parent / "data" / "raw"
OUT_PLOTS = Path(__file__).parent.parent / "results" / "plots"
OUT_TABLE = Path(__file__).parent.parent / "results"

GPT4 = pd.Timestamp("2023-03-14")
WINDOW_START = pd.Timestamp("2022-09-14")
WINDOW_END = pd.Timestamp("2023-09-14")  # 7 months pre, 6 months post; pre-Oct-7

TECH = {"NICE.TA", "XLK", "IXN", "QQQ"}
NON_TECH = {"TA35.TA", "EWU", "EWG", "EWS", "VGK"}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= WINDOW_START) & (df["date"] <= WINDOW_END)].copy()
    df = df[df["ticker"].isin(TECH | NON_TECH)].copy()
    df["log_close"] = np.log(df["close"])
    df["tech"] = df["ticker"].isin(TECH).astype(int)
    df["post"] = (df["date"] >= GPT4).astype(int)
    df["tech_post"] = df["tech"] * df["post"]

    # Per-ticker rebase to log(0) at WINDOW_START so different price magnitudes
    # don't dominate the regression
    out = []
    for _, g in df.groupby("ticker"):
        g = g.sort_values("date").copy()
        g["log_close_norm"] = g["log_close"] - g["log_close"].iloc[0]
        out.append(g)
    return pd.concat(out, ignore_index=True)


def fit_did(df: pd.DataFrame):
    X = df[["tech", "post", "tech_post"]]
    X = sm.add_constant(X)
    y = df["log_close_norm"]
    model = sm.OLS(y, X, hasconst=True).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["ticker"]},
    )
    return model


def plot_parallel_trends(df: pd.DataFrame) -> None:
    """Average each group over time and plot. Helps eyeball the parallel-trends assumption."""
    grp = df.groupby(["date", "tech"])["log_close_norm"].mean().reset_index()
    pivot = grp.pivot(index="date", columns="tech", values="log_close_norm")
    pivot.columns = ["non_tech", "tech"]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(pivot.index, pivot["tech"], label="tech (NICE, XLK, IXN, QQQ)",
            color="purple", linewidth=2)
    ax.plot(pivot.index, pivot["non_tech"], label="non-tech (TA-35, EWU, EWG, EWS, VGK)",
            color="darkgray", linewidth=2, linestyle="--")
    ax.axvline(GPT4, color="red", linestyle=":", linewidth=1.5)
    ax.text(GPT4, ax.get_ylim()[1] * 0.97, " GPT-4 release",
            color="red", fontsize=10, verticalalignment="top")

    ax.set_title("DiD: tech vs non-tech log-prices around GPT-4 release\n(rebased to 0 at window start; group averages)")
    ax.set_ylabel("log(close) - log(close at window start)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = OUT_PLOTS / "did_ai_parallel_trends.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"saved {out}")


def main() -> None:
    df = pd.read_csv(DATA / "tase_daily.csv", parse_dates=["date"])
    df = prepare(df)
    model = fit_did(df)

    print(model.summary().tables[1])
    print()

    beta = model.params["tech_post"]
    ci_lo, ci_hi = model.conf_int(alpha=0.05).loc["tech_post"]
    pct = (np.exp(beta) - 1) * 100
    pct_lo = (np.exp(ci_lo) - 1) * 100
    pct_hi = (np.exp(ci_hi) - 1) * 100

    print(f"DiD effect on tech vs non-tech after GPT-4: "
          f"{pct:+.2f}% [{pct_lo:+.2f}%, {pct_hi:+.2f}%], "
          f"p = {model.pvalues['tech_post']:.4f}")

    out_df = pd.DataFrame([{
        "treatment_date": GPT4.date(),
        "window_start": WINDOW_START.date(),
        "window_end": WINDOW_END.date(),
        "n_tech_tickers": len(TECH),
        "n_non_tech_tickers": len(NON_TECH),
        "n_observations": len(df),
        "did_coefficient": round(beta, 4),
        "did_pct_effect": round(pct, 2),
        "did_pct_ci95_lo": round(pct_lo, 2),
        "did_pct_ci95_hi": round(pct_hi, 2),
        "p_value": round(model.pvalues["tech_post"], 4),
        "r_squared": round(model.rsquared, 3),
    }])
    out_path = OUT_TABLE / "did_ai_results.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nsaved {out_path}")

    plot_parallel_trends(df)


if __name__ == "__main__":
    main()
