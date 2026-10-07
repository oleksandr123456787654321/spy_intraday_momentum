"""
data.py - download and clean 1-MINUTE SPY bars from Alpaca (free tier, IEX feed).

Usage:
    pip install alpaca-py pandas numpy
    export ALPACA_KEY=...
    export ALPACA_SECRET=...
    python data.py

Output: spy_1min.csv
    index   = New York time (bar START timestamp, 09:30 ... 15:59)
    columns = open, high, low, close, volume, filled
    filled  = True if the minute had no trades and was filled from the previous close
"""
import os
import sys
import time as _time
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOL = "SPY"
YEARS = 7
CHUNK_DAYS = 60                 # download in chunks, each saved to disk
CACHE_DIR = Path("cache_1min")  # re-running skips chunks already downloaded
OUT_FILE = "spy_1min.csv"
TZ = "America/New_York"
MAX_MISSING_FRAC = 0.10         # drop a day if >10% of its minutes had no trades
RETRIES = 3


# ----------------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------------
def download(key, secret, start, end):
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(key, secret)
    CACHE_DIR.mkdir(exist_ok=True)
    frames = []
    s = start
    while s < end:
        e = min(s + pd.Timedelta(days=CHUNK_DAYS), end)
        cache_file = CACHE_DIR / f"{SYMBOL}_{s.date()}_{e.date()}.csv"

        if cache_file.exists():
            part = pd.read_csv(cache_file, index_col=0)
            part.index = pd.to_datetime(part.index, utc=True)
            print(f"cached     {s.date()} -> {e.date()}: {len(part)} bars")
        else:
            part = None
            for attempt in range(1, RETRIES + 1):
                try:
                    req = StockBarsRequest(
                        symbol_or_symbols=SYMBOL,
                        timeframe=TimeFrame.Minute,
                        start=s.to_pydatetime(),
                        end=e.to_pydatetime(),
                        adjustment=Adjustment.RAW,
                        feed=DataFeed.IEX,
                    )
                    bars = client.get_stock_bars(req).df
                    part = bars.droplevel("symbol") if len(bars) else pd.DataFrame()
                    break
                except Exception as ex:  # network / rate limit
                    print(f"  attempt {attempt} failed: {ex}")
                    _time.sleep(5 * attempt)
            if part is None:
                sys.exit(f"Failed to download {s.date()} -> {e.date()}. Re-run to resume.")
            part.to_csv(cache_file)
            print(f"downloaded {s.date()} -> {e.date()}: {len(part)} bars")

        if len(part):
            frames.append(part)
        s = e

    if not frames:
        sys.exit("No data returned at all. Check keys / plan limits.")
    return pd.concat(frames)


# ----------------------------------------------------------------------------
# Clean
# ----------------------------------------------------------------------------
def clean(df):
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df.index = df.index.tz_convert(TZ)
    df = df.between_time("09:30", "15:59")
    df = df[["open", "high", "low", "close", "volume"]]

    days_out = []
    dropped_short, dropped_gaps = [], []
    for date, g in df.groupby(df.index.date):
        # (3) full session required
        if g.index[0].time() > time(9, 35) or g.index[-1].time() < time(15, 55):
            dropped_short.append(date)
            continue

        # (4) rebuild the full 390-minute grid and fill empty minutes
        grid = pd.date_range(f"{date} 09:30", f"{date} 15:59", freq="1min", tz=TZ)
        g = g.reindex(grid)
        missing = g["close"].isna()

        # (5) too many empty minutes -> unreliable day
        if missing.mean() > MAX_MISSING_FRAC:
            dropped_gaps.append(date)
            continue

        g["close"] = g["close"].ffill().bfill()
        for col in ("open", "high", "low"):
            g.loc[missing, col] = g.loc[missing, "close"]
        g["volume"] = g["volume"].fillna(0)
        g["filled"] = missing
        days_out.append(g)

    print(f"\nDays kept:                         {len(days_out)}")
    print(f"Dropped (not a full session):      {len(dropped_short)}")
    print(f"Dropped (> {MAX_MISSING_FRAC:.0%} missing minutes):    {len(dropped_gaps)}")
    if dropped_gaps:
        print("  gap days (first 10):", dropped_gaps[:10])

    out = pd.concat(days_out)

    # sanity checks
    real = out[~out["filled"]]
    assert (real["high"] >= real["low"]).all(), "high < low somewhere"
    assert (out[["open", "high", "low", "close"]] > 0).all().all(), "non-positive price"
    assert out.groupby(out.index.date).size().eq(390).all(), "a kept day does not have 390 minutes"
    return out


# ----------------------------------------------------------------------------
# Helpers for later steps
# ----------------------------------------------------------------------------
def load_minute(path=OUT_FILE):
    """Load the cleaned 1-minute file with a proper New York DatetimeIndex."""
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(TZ)
    return df


def to_halfhour(df):
    """Aggregate 1-minute bars to 30-minute bars (labeled by START time, 09:30 ... 15:30)."""
    return df.resample("30min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open"])


# ----------------------------------------------------------------------------
def main():
    key, secret = os.environ.get("ALPACA_KEY"), os.environ.get("ALPACA_SECRET")
    if not key or not secret:
        sys.exit("Set ALPACA_KEY and ALPACA_SECRET environment variables first.")

    end = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=2)
    start = end - pd.DateOffset(years=YEARS)

    raw = download(key, secret, start, end)

    first = raw.index.min()
    print(f"\nEarliest bar returned: {first}")
    if first > start + pd.Timedelta(days=30):
        print(f"WARNING: you asked for data from {start.date()} but history only starts "
              f"{first.date()}. Your plan likely limits history depth.")

    df = clean(raw)
    df.to_csv(OUT_FILE)

    n_days = df.index.normalize().nunique()
    print(f"\nSaved {len(df):,} bars ({n_days} days) to {OUT_FILE}")
    print(f"Range: {df.index[0]}  ->  {df.index[-1]}")
    print(f"Filled (no-trade) minutes: {df['filled'].mean():.2%}")


if __name__ == "__main__":
    main()