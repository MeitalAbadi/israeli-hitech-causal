"""
Pull Tel Aviv Stock Exchange index data as a proxy for Israeli hi-tech sector health.

We use yfinance because it's free, reliable, and the TASE indices are listed there.

Indices we want:
- TA-35   : top 35 companies (broad market)
- TA-125  : broader Tel Aviv Stock Exchange index
- TA-90   : mid-cap (excludes the top 35)

We don't have a clean tech-only TASE index on yfinance, so we compose a basket
of the largest Israeli tech companies on TASE (e.g. Nice, Check Point,
Plus500, Allot) and treat that as our hi-tech proxy.

Output: data/raw/tase_daily.csv with columns date, ticker, close, volume.
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime

OUT = Path(__file__).parent.parent / "data" / "raw" / "tase_daily.csv"

START = "2021-01-01"
END = datetime.now().strftime("%Y-%m-%d")

INDICES = {
    "TA35.TA": "TA-35",
    "TA90.TA": "TA-90",
    "TA125.TA": "TA-125",
}

# Hi-tech basket: large-cap Israeli tech listed on TASE.
# Not exhaustive, but a reasonable index of Israeli tech sentiment.
TECH_BASKET = {
    "NICE.TA":   "Nice",
    "ALLT.TA":   "Allot",
    "FORTY.TA":  "Formula Systems",
}

# International comparison indices for synthetic control + DiD analysis.
# Treatment unit (Israel) needs to be compared to non-affected comparable markets.
COMPARISON = {
    "QQQ":  "NASDAQ-100 ETF (US tech)",
    "XLK":  "S&P 500 Tech Sector ETF",
    "IXN":  "iShares Global Tech ETF",
    "EWU":  "iShares UK ETF",
    "EWG":  "iShares Germany ETF",
    "EWS":  "iShares Singapore ETF",
    "VGK":  "Vanguard FTSE Europe ETF",
}


def pull(ticker: str, name: str) -> pd.DataFrame:
    df = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df["ticker"] = ticker
    df["name"] = name
    return df[["Date", "ticker", "name", "Close", "Volume"]].rename(
        columns={"Date": "date", "Close": "close", "Volume": "volume"}
    )


def main() -> None:
    frames = []
    all_tickers = {**INDICES, **TECH_BASKET, **COMPARISON}
    for ticker, name in all_tickers.items():
        print(f"  fetching {ticker} ({name})...")
        df = pull(ticker, name)
        if df.empty:
            print(f"    nothing returned for {ticker}, skipping")
            continue
        frames.append(df)

    if not frames:
        raise SystemExit("no data fetched, can't continue")

    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Tag each ticker with its group (israel vs comparison) for downstream analysis
    israel_tickers = set(INDICES.keys()) | set(TECH_BASKET.keys())
    out["group"] = out["ticker"].apply(
        lambda t: "israel" if t in israel_tickers else "comparison"
    )

    out.to_csv(OUT, index=False)
    print(f"saved {len(out)} rows to {OUT}")
    print(f"  israel rows: {(out['group']=='israel').sum()}")
    print(f"  comparison rows: {(out['group']=='comparison').sum()}")


if __name__ == "__main__":
    main()
