"""
switch_vix.py - choose the stop rule each day from the VIX.

The idea
    We have two exit rules with identical entry signals:
        Baseline A : loose stop (opposite band)  - lets trends run, suffers late reversals
        Band+VWAP  : tight stop (band or VWAP)   - cuts losses, but also cuts big winners
    Each morning the strategy looks at the VIX (the market's expected volatility) and picks one.

The rule
    * Loose stop (Baseline A) when yesterday's VIX closed ABOVE VIX_THRESHOLD.
    * Tight stop (Band+VWAP) otherwise.
    Everything else (bands, entries, sizing, costs, train/test split) is exactly as in main.py.

Why this direction and this threshold
    * Threshold 20: the conventional boundary between a "normal" and an "elevated" VIX.
      It is a round number from market convention, not tuned on this sample.
    * Direction: the paper (Section 4.1) finds the strategy works better when the VIX is high,
      i.e. trends are larger and more persistent, so there the stop should let trends run.
      When the VIX is low, moves are small and breakouts fade more often, so cutting losses
      early should help.

Run:  pip install yfinance
      python switch_vix.py      (needs main.py, data.py and spy_1min.csv in the same folder)
The VIX is downloaded once and cached in vix_daily.csv. Output goes to ./results_switch_vix/
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from main import (LOOKBACK, TRAIN_END, generate_trades, leverage_by_day, load_market, noise_area,
                  print_report, run_account, vwap_at_marks)

# =============================================================================
# Settings (decided in advance, see the docstring)
# =============================================================================
VIX_THRESHOLD = 20.0
LOOSE_WHEN = "above"           # "above": loose stop when VIX > threshold, tight stop otherwise
VIX_FILE = "vix_daily.csv"
OUT_DIR = Path("results_switch_vix")

COLOR = {"Baseline A": "black", "Band+VWAP": "green", "Band+VWAP Dyn": "blue",
         "Switch VIX": "orange", "Switch VIX Dyn": "gold", "SPY buy&hold": "red"}


# =============================================================================
# VIX data
# =============================================================================
def download_vix(first_date, last_date):
    import yfinance as yf                                    # imported here so the rest works without it
    raw = yf.download("^VIX", start=(first_date - pd.Timedelta(days=14)).date(),
                      end=(last_date + pd.Timedelta(days=5)).date(),
                      auto_adjust=False, progress=False)
    if raw.empty:
        raise SystemExit("yfinance returned no VIX data. Check your internet connection and try again.")
    if isinstance(raw.columns, pd.MultiIndex):               # newer yfinance versions return 2-level columns
        raw.columns = raw.columns.get_level_values(0)
    vix = raw["Close"].rename("close")
    vix.index = pd.to_datetime(vix.index).tz_localize(None).normalize()
    return vix.dropna().sort_index()


def load_vix(first_date, last_date):
    """Daily VIX close, from the cache file if it covers our dates, otherwise from yfinance."""
    if Path(VIX_FILE).exists():
        vix = pd.read_csv(VIX_FILE, index_col=0, parse_dates=True)["close"].dropna().sort_index()
        if vix.index.min() <= first_date and vix.index.max() >= last_date - pd.Timedelta(days=7):
            return vix
        print("Cached VIX file does not cover the SPY dates, downloading again.")
    vix = download_vix(first_date, last_date)
    vix.to_frame().to_csv(VIX_FILE)
    return vix


def vix_before_each_day(vix, dates):
    """For each SPY day: the VIX close of the last trading day strictly BEFORE it (known at the open)."""
    pos = vix.index.searchsorted(dates, side="left") - 1
    out = np.full(len(dates), np.nan)
    ok = pos >= 0
    out[ok] = vix.to_numpy()[pos[ok]]
    lag_days = np.full(len(dates), np.nan)
    lag_days[ok] = (dates[ok] - vix.index[pos[ok]]).days
    return out, lag_days


# =============================================================================
# The switching rule
# =============================================================================
def choose_stop(vix_prev, threshold=VIX_THRESHOLD, loose_when=LOOSE_WHEN):
    """
    True on days that use the tight stop (Band+VWAP), False on days that use the loose stop (Baseline A).
    Days without a VIX value default to the tight stop.
    """
    use_tight = np.ones(len(vix_prev), dtype=bool)
    ok = ~np.isnan(vix_prev)
    above = vix_prev[ok] > threshold
    use_tight[ok] = ~above if loose_when == "above" else above
    return use_tight


def build_switch_trades(trades_tight, trades_loose, use_tight):
    """
    Take each day's trades from the tight-stop list or the loose-stop list. This is exact because
    signals never depend on position size: only the account (run_account) compounds across days.
    """
    tight = trades_tight[use_tight[trades_tight["d"].to_numpy()]]
    loose = trades_loose[~use_tight[trades_loose["d"].to_numpy()]]
    return pd.concat([tight, loose]).sort_values("d", kind="stable").reset_index(drop=True)


# =============================================================================
# Plots
# =============================================================================
def save_plots(runs, spy, table):
    equity = {name: (1 + daily["ret"]).cumprod() for name, (_, daily) in runs.items()}
    equity["SPY buy&hold"] = (1 + spy.fillna(0)).cumprod()
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, eq in equity.items():
        ax.plot(eq, label=name, color=COLOR[name], ls="--" if name == "Switch VIX Dyn" else "-")
    ax.axvline(pd.Timestamp(TRAIN_END), color="gray", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_title("Equity curves with the VIX switch (log scale, start = 1; dashed line = start of test)")
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "equity_curves_switch_vix.png", dpi=150)
    plt.close(fig)

    order = list(COLOR)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (col, title) in zip(axes, [("sharpe", "Sharpe ratio"), ("ann_ret", "Annualized return"),
                                       ("ann_vol", "Annualized volatility")]):
        t = table[col].unstack(level=1).loc[["Train", "Test"]][order]
        t.plot.bar(ax=ax, rot=0, color=[COLOR[n] for n in order], legend=(col == "sharpe"))
        ax.set_title(title)
        ax.set_xlabel("")
        ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "metrics_comparison_switch_vix.png", dpi=150)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main():
    OUT_DIR.mkdir(exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    m = load_market()
    bands, vwap_mark, lev = noise_area(m), vwap_at_marks(m), leverage_by_day(m)

    # the two existing rule sets (identical to main.py)
    trades_a = generate_trades(m, bands, vwap_mark, "A")
    trades_v = generate_trades(m, bands, vwap_mark, "band_vwap")

    # VIX known at each day's open
    vix = load_vix(m["dates"][0], m["dates"][-1])
    vix_prev, lag = vix_before_each_day(vix, m["dates"])
    print(f"VIX data: {len(vix)} days, {vix.index[0].date()} -> {vix.index[-1].date()} | "
          f"SPY days without a VIX value: {int(np.isnan(vix_prev).sum())} | "
          f"longest gap between a SPY day and the VIX date used: {np.nanmax(lag):.0f} days")

    # the switch: no fitted numbers
    use_tight = choose_stop(vix_prev)
    trades_s = build_switch_trades(trades_v, trades_a, use_tight)
    print(f"\nRULE (fixed in advance): loose stop (Baseline A) when yesterday's VIX closed above "
          f"{VIX_THRESHOLD:.0f}, tight stop (Band+VWAP) otherwise")

    d_dates = m["dates"][LOOKBACK:]
    loose = ~use_tight[LOOKBACK:]
    print(f"Days using the loose stop: train {loose[d_dates <= TRAIN_END].mean():.0%} "
          f"({int(loose[d_dates <= TRAIN_END].sum())} days) | "
          f"test {loose[d_dates > TRAIN_END].mean():.0%} ({int(loose[d_dates > TRAIN_END].sum())} days)")

    runs = {
        "Baseline A": (trades_a, run_account(m, trades_a)),
        "Band+VWAP": (trades_v, run_account(m, trades_v)),
        "Band+VWAP Dyn": (trades_v, run_account(m, trades_v, leverage=lev)),
        "Switch VIX": (trades_s, run_account(m, trades_s)),
        "Switch VIX Dyn": (trades_s, run_account(m, trades_s, leverage=lev)),
    }
    first_day = runs["Baseline A"][1].index
    spy = pd.Series(m["C"][:, -1], index=m["dates"]).pct_change().loc[first_day]

    table = print_report(runs, spy)
    save_plots(runs, spy, table)

    table.to_csv(OUT_DIR / "summary_switch_vix.csv")
    trades_s.to_csv(OUT_DIR / "trades_Switch_VIX.csv", index=False)
    runs["Switch VIX"][1].to_csv(OUT_DIR / "daily_Switch_VIX.csv")
    runs["Switch VIX Dyn"][1].to_csv(OUT_DIR / "daily_Switch_VIX_Dyn.csv")
    print(f"\nPlots and CSVs saved in {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()