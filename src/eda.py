"""
Exploratory analysis of TASE indices around Iron Swords (2023-10-07)
and around major AI tool releases.

Goal: see whether the time series visibly shifts around the events of
interest. We don't claim causality here — that's what the ITS model
does next. This is just looking at the raw data with the event lines
drawn on top.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path(__file__).parent.parent / "data" / "raw"
OUT = Path(__file__).parent.parent / "results" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

OCT7 = pd.Timestamp("2023-10-07")
GPT4 = pd.Timestamp("2023-03-14")
CHATGPT = pd.Timestamp("2022-11-30")
DEVIN = pd.Timestamp("2024-03-12")


def load_prices() -> pd.DataFrame:
    df = pd.read_csv(DATA / "tase_daily.csv", parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"])
    return df[df["date"] >= "2022-01-01"].copy()


def normalize_to_100(df: pd.DataFrame, base_date: str = "2023-01-01") -> pd.DataFrame:
    """Rebase each ticker so base_date = 100, for visual comparability across magnitudes."""
    out = []
    base_ts = pd.Timestamp(base_date)
    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("date").copy()
        base_row = g[g["date"] >= base_ts].head(1)
        if base_row.empty:
            continue
        base_val = base_row["close"].iloc[0]
        if base_val == 0 or pd.isna(base_val):
            continue
        g["normalized"] = g["close"] / base_val * 100
        out.append(g)
    return pd.concat(out, ignore_index=True)


def plot_israel_vs_world(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5))

    israel = df[df["group"] == "israel"]
    for _, g in israel.groupby("ticker"):
        ax.plot(g["date"], g["normalized"],
                label=f"IL: {g['name'].iloc[0]}",
                alpha=0.85, linewidth=1.6)

    comp = df[df["group"] == "comparison"]
    for _, g in comp.groupby("ticker"):
        ax.plot(g["date"], g["normalized"],
                label=f"world: {g['name'].iloc[0]}",
                alpha=0.4, linestyle="--", linewidth=1.0)

    y_top = ax.get_ylim()[1]
    for ts, label, color in [
        (CHATGPT, "ChatGPT release", "gray"),
        (GPT4, "GPT-4", "purple"),
        (OCT7, "Iron Swords", "red"),
        (DEVIN, "Devin", "orange"),
    ]:
        ax.axvline(ts, color=color, linestyle=":", linewidth=1.5, alpha=0.7)
        ax.text(ts, y_top * 0.97, f" {label}", color=color, fontsize=9, rotation=0,
                verticalalignment="top")

    ax.set_title(
        "Israeli stock indices vs international comparison, 2022 to 2026\n"
        "(rebased to 100 at 2023-01-01; observational only, not yet causal)",
        fontsize=12,
    )
    ax.set_ylabel("Normalized close (100 = 2023-01-01)")
    ax.legend(loc="best", fontsize=8, ncol=2, framealpha=0.85)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = OUT / "01_israel_vs_world.png"
    plt.savefig(out_path, dpi=120)
    print(f"saved {out_path}")
    plt.close()


def plot_war_zoom(df: pd.DataFrame) -> None:
    """Zoom in to 6 months around Oct 7 for a closer look at the war shock."""
    window_start = OCT7 - pd.Timedelta(days=180)
    window_end = OCT7 + pd.Timedelta(days=180)
    sub = df[(df["date"] >= window_start) & (df["date"] <= window_end)].copy()
    sub = normalize_to_100(sub, base_date=window_start.strftime("%Y-%m-%d"))

    fig, ax = plt.subplots(figsize=(13, 6))

    israel = sub[sub["group"] == "israel"]
    for _, g in israel.groupby("ticker"):
        ax.plot(g["date"], g["normalized"], label=f"IL: {g['name'].iloc[0]}",
                alpha=0.85, linewidth=1.8)

    comp = sub[sub["group"] == "comparison"]
    for _, g in comp.groupby("ticker"):
        ax.plot(g["date"], g["normalized"], label=f"world: {g['name'].iloc[0]}",
                alpha=0.35, linestyle="--", linewidth=1.0)

    ax.axvline(OCT7, color="red", linestyle="-", linewidth=2, alpha=0.8)
    ax.text(OCT7, ax.get_ylim()[1] * 0.97, " Iron Swords (Oct 7, 2023)",
            color="red", fontsize=10, fontweight="bold", verticalalignment="top")

    ax.set_title("Six months around October 7, 2023\n(rebased to 100 at window start)", fontsize=12)
    ax.set_ylabel("Normalized close")
    ax.legend(loc="best", fontsize=8, ncol=2, framealpha=0.85)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = OUT / "02_war_zoom.png"
    plt.savefig(out_path, dpi=120)
    print(f"saved {out_path}")
    plt.close()


def quick_descriptives(df: pd.DataFrame) -> None:
    """Print a few summary numbers so we know what we're working with."""
    print("\n=== Descriptives ===")
    print(f"date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"tickers: {df['ticker'].nunique()}")
    print("\nrows per ticker:")
    print(df.groupby(["group", "ticker", "name"]).size().to_string())


def main() -> None:
    df = load_prices()
    df = normalize_to_100(df, base_date="2023-01-01")
    quick_descriptives(df)
    plot_israel_vs_world(df)
    plot_war_zoom(df)


if __name__ == "__main__":
    main()
