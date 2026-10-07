"""
main.py - Intraday momentum on SPY: replication of Zarattini, Aziz & Barbon (2025).

What this script does
    1. Loads the cleaned 1-minute SPY file produced by data.py.
    2. Builds the "Noise Area": bands around the open that show how far price normally
       travels at each time of day (average of the previous 14 days).
    3. Runs three versions of the strategy on identical data and identical costs:
         - Baseline A     : stop = the opposite band, 100% of equity, no leverage
         - Band+VWAP      : stop = current band or VWAP (tighter), same sizing     <- extension
         - Band+VWAP Dyn  : same signals as Band+VWAP, but position size targets 2% daily
                            volatility (leverage capped at 4x), as in the paper's final model
    4. Reports annualized return, annualized volatility and Sharpe for train and test
       periods next to SPY buy-and-hold, and saves plots to ./results/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # write plots to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import load_minute

# =============================================================================
# Settings
# =============================================================================
DATA_FILE = "spy_1min.csv"
RESULTS_DIR = Path("results")

LOOKBACK = 14                  # days used for the Noise Area and for recent volatility (paper's value)
COMMISSION = 0.0035            # $ per share (Interactive Brokers entry-level rate)
SLIPPAGE = 0.001               # $ per share (paper's live measurement)
INIT_CASH = 100_000.0
TRAIN_END = "2023-12-31"       # fixed BEFORE looking at results: train = start..2023, test = 2024..

TARGET_VOL = 0.02              # dynamic sizing: aim for 2% daily volatility (paper's value)
MAX_LEVERAGE = 4.0             # dynamic sizing: leverage capped at 4x (paper's value)

SHOW_DIAGNOSTICS = False       # extra tables: by year, long vs short, biggest days

MINUTES = 390
MARKS = np.arange(30, MINUTES, 30)                                  # 10:00 ... 15:30 (12 marks)
LABELS = [f"{(570 + m) // 60:02d}:{(570 + m) % 60:02d}" for m in MARKS]


# =============================================================================
# 1. Data
# =============================================================================
def load_market():
    """Reshape the 1-minute table into (n_days, 390) arrays: O/H/L/C/V[d, minute]."""
    df = load_minute(DATA_FILE).sort_index()
    assert len(df) % MINUTES == 0, "some day does not have exactly 390 minutes"
    assert (df.index[::MINUTES].strftime("%H:%M") == "09:30").all(), "a day does not start at 09:30"
    n_days = len(df) // MINUTES

    def grid(col):
        return df[col].to_numpy().reshape(n_days, MINUTES)

    return {
        "O": grid("open"), "H": grid("high"), "L": grid("low"), "C": grid("close"), "V": grid("volume"),
        "dates": pd.DatetimeIndex(df.index[::MINUTES].normalize().tz_localize(None)),
        "n_days": n_days,
    }


# =============================================================================
# 2. Noise Area, VWAP and leverage
# =============================================================================
def noise_area(m):
    """
    sigma[d, k] = average of |price at mark k / day open - 1| over the previous LOOKBACK days.
    Bands are built around max/min(open, previous close) so overnight gaps count as imbalance.
    """
    O, C, n = m["O"], m["C"], m["n_days"]

    moves = np.abs(C[:, MARKS - 1] / O[:, [0]] - 1)          # today's move from the open at each mark
    sigma = np.full(moves.shape, np.nan)
    for d in range(LOOKBACK, n):
        sigma[d] = moves[d - LOOKBACK:d].mean(axis=0)        # previous days only, never day d itself

    prev_close = np.full(n, np.nan)
    prev_close[1:] = C[:-1, -1]

    upper = np.maximum(O[:, 0], prev_close)[:, None] * (1 + sigma)
    lower = np.minimum(O[:, 0], prev_close)[:, None] * (1 - sigma)
    return {"sigma": sigma, "upper": upper, "lower": lower, "prev_close": prev_close}


def vwap_at_marks(m):
    """Volume-weighted average price since 09:30, as known at each decision mark."""
    typical = (m["H"] + m["L"] + m["C"]) / 3
    cum_vol = np.cumsum(m["V"], axis=1)
    vwap = np.cumsum(typical * m["V"], axis=1) / np.where(cum_vol > 0, cum_vol, np.nan)
    vwap = pd.DataFrame(vwap).ffill(axis=1).to_numpy()      # minutes with no volume keep the last value
    return vwap[:, MARKS - 1]


def leverage_by_day(m):
    """
    Exposure multiplier for each day = TARGET_VOL / recent SPY daily volatility, capped at MAX_LEVERAGE.
    Recent volatility = std of the LOOKBACK daily returns before day d (shift(1) keeps today out of it).
    Where it is not available yet (first days), leverage defaults to 1.
    """
    close = pd.Series(m["C"][:, -1])
    recent_vol = close.pct_change().rolling(LOOKBACK).std().shift(1)
    leverage = (TARGET_VOL / recent_vol).clip(upper=MAX_LEVERAGE)
    return leverage.fillna(1.0).to_numpy()


# =============================================================================
# 3. Trading rules
# =============================================================================
def generate_trades(m, bands, vwap_mark, mode):
    """
    mode "A"         : enter on a band breakout, exit only when price crosses the OPPOSITE band
    mode "band_vwap" : enter on a breakout beyond max(upper, VWAP) / min(lower, VWAP);
                       exit as soon as price falls back below that same level
    In both modes a stop-out can flip straight into the opposite position, and everything
    open is closed at the end of the day.
    """
    O, C = m["O"], m["C"]
    rows = []
    for d in range(LOOKBACK, m["n_days"]):
        pos, entry_px, entry_time = 0, np.nan, ""            # pos: 0 flat, 1 long, -1 short
        for k, minute in enumerate(MARKS):
            price = C[d, minute - 1]                         # last known price
            ub, lb, vw = bands["upper"][d, k], bands["lower"][d, k], vwap_mark[d, k]

            if mode == "A":
                long_level, short_level = ub, lb
                stop_long, stop_short = price < lb, price > ub
            else:
                long_level, short_level = np.fmax(ub, vw), np.fmin(lb, vw)
                stop_long, stop_short = price < long_level, price > short_level

            target = pos
            if (pos == 1 and stop_long) or (pos == -1 and stop_short):
                target = 0                                   # stopped out
            if target == 0:                                  # flat (or just stopped): look for an entry
                if price > long_level:
                    target = 1
                elif price < short_level:
                    target = -1

            if target != pos:
                fill = O[d, minute]                          # fill at the open of the mark minute
                if pos != 0:
                    reason = "stop" if target == 0 else "reverse"
                    rows.append((d, pos, entry_time, entry_px, LABELS[k], fill, reason))
                pos, entry_px, entry_time = target, fill, LABELS[k]

        if pos != 0:                                         # end-of-day exit
            rows.append((d, pos, entry_time, entry_px, "16:00", C[d, -1], "close"))

    trades = pd.DataFrame(rows, columns=["d", "pos", "entry_time", "entry_px", "exit_time", "exit_px", "reason"])
    trades["date"] = m["dates"][trades["d"].to_numpy()]
    trades["pnl_per_share"] = trades["pos"] * (trades["exit_px"] - trades["entry_px"])
    return trades


# =============================================================================
# 4. Sizing, costs, equity
# =============================================================================
def run_account(m, trades, leverage=None, commission=COMMISSION, slippage=SLIPPAGE):
    """
    Position size = yesterday's equity x leverage / today's open (leverage = 1 if not given).
    Equity compounds daily; costs are charged on every execution (entry and exit).
    """
    O = m["O"]
    per_day = {d: g for d, g in trades.groupby("d")}
    aum, out = INIT_CASH, []
    for d in range(LOOKBACK, m["n_days"]):
        aum_prev, gross, costs, shares = aum, 0.0, 0.0, 0
        if d in per_day:
            g = per_day[d]
            lev = 1.0 if leverage is None else leverage[d]
            shares = int(aum_prev * lev // O[d, 0])
            gross = (g["pnl_per_share"] * shares).sum()
            costs = len(g) * shares * 2 * (commission + slippage)       # one entry + one exit per trade
        pnl = gross - costs
        aum = aum_prev + pnl
        out.append((m["dates"][d], shares, gross, costs, pnl, aum, pnl / aum_prev))
    cols = ["date", "shares", "gross", "costs", "pnl", "aum", "ret"]
    return pd.DataFrame(out, columns=cols).set_index("date")


# =============================================================================
# 5. Performance numbers
# =============================================================================
def metrics(r):
    r = r.dropna()
    n = len(r)
    equity = (1 + r).cumprod()
    ann_ret = equity.iloc[-1] ** (252 / n) - 1
    vol = r.std() * np.sqrt(252)
    return pd.Series({
        "days": n,
        "total_ret": equity.iloc[-1] - 1,
        "ann_ret": ann_ret,
        "ann_vol": vol,
        "sharpe": ann_ret / vol,
        "max_dd": (equity / equity.cummax() - 1).min(),
        "hit": (r[r != 0] > 0).mean(),               # winning days / days with a position
        "skew": r.skew(),
    })


def alpha_beta(y, x):
    """Regress strategy daily returns y on SPY daily returns x. Returns annualized alpha, beta, alpha t-stat."""
    d = pd.concat([y, x], axis=1).dropna()
    d.columns = ["y", "x"]
    n = len(d)
    beta = np.cov(d.x, d.y)[0, 1] / d.x.var()
    alpha = d.y.mean() - beta * d.x.mean()
    resid = d.y - alpha - beta * d.x
    se = np.sqrt(resid.var(ddof=2) * (1 / n + d.x.mean() ** 2 / ((d.x - d.x.mean()) ** 2).sum()))
    return alpha * 252, beta, alpha / se


def period_masks(index):
    return {"Train": index <= TRAIN_END, "Test": index > TRAIN_END, "Full": np.ones(len(index), bool)}


# =============================================================================
# 6. Reporting
# =============================================================================
def print_report(runs, spy):
    idx = next(iter(runs.values()))[1].index
    masks = period_masks(idx)

    # main table
    rows = {}
    for period, mask in masks.items():
        for name, (_, daily) in runs.items():
            rows[(period, name)] = metrics(daily["ret"][mask])
        rows[(period, "SPY buy&hold")] = metrics(spy[mask])
    table = pd.DataFrame(rows).T
    print("\nPERFORMANCE (returns are per year, Sharpe = return / volatility)")
    print(table.round(3).to_string())

    # alpha / beta
    print("\nALPHA (annualized) / BETA / alpha t-stat, regression on SPY daily returns")
    for name, (_, daily) in runs.items():
        for period, mask in masks.items():
            a, b, t = alpha_beta(daily["ret"][mask], spy[mask])
            print(f"  {name:14s} {period:5s} alpha {a:7.3f} | beta {b:6.3f} | t {t:5.2f}")

    # trade statistics (Band+VWAP and Band+VWAP Dyn share the same trades; only the size differs)
    n_days = len(idx)
    stats = {}
    for name, (trades, daily) in runs.items():
        g, c = daily["gross"].sum(), daily["costs"].sum()
        stats[name] = {
            "trades": len(trades), "per_day": len(trades) / n_days,
            "win_rate": (trades["pnl_per_share"] > 0).mean(),
            "gross_$": g, "costs_$": c, "costs/gross": c / g if g > 0 else np.nan,
            "final_equity": daily["aum"].iloc[-1],
        }
    print("\nTRADE STATISTICS (full sample)")
    print(pd.DataFrame(stats).round(3).to_string())
    return table


def cost_sensitivity(m, runs, lev):
    """Re-run the accounting with worse and worse slippage (commission fixed). Signals do not change."""
    levels = [0.0, 0.001, 0.0025, 0.005, 0.01, 0.02]          # $ per share, per execution
    rows = []
    for s in levels:
        for name, (trades, _) in runs.items():
            r = run_account(m, trades, leverage=lev if "Dyn" in name else None, slippage=s)["ret"]
            for period, mask in period_masks(r.index).items():
                rows.append({"slippage": s, "strategy": name, "period": period,
                             "sharpe": metrics(r[mask])["sharpe"]})
    out = pd.DataFrame(rows)
    print("\nSHARPE vs SLIPPAGE ($/share per execution; base case in the paper = 0.001)")
    print(out.pivot_table(index="slippage", columns=["strategy", "period"], values="sharpe").round(2).to_string())


def save_plots(runs, spy, table):
    equity = {name: (1 + daily["ret"]).cumprod() for name, (_, daily) in runs.items()}
    equity["SPY buy&hold"] = (1 + spy.fillna(0)).cumprod()

    fig, ax = plt.subplots(figsize=(10, 5))
    for (name, eq), color in zip(equity.items(), ["black", "green", "blue", "red"]):
        ax.plot(eq, label=name, color=color)
    ax.axvline(pd.Timestamp(TRAIN_END), color="k", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_title("Equity curves (log scale, start = 1; dashed line = start of test period)")
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "equity_curves.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (col, title) in zip(axes, [("sharpe", "Sharpe ratio"), ("ann_ret", "Annualized return"),
                                       ("ann_vol", "Annualized volatility")]):
        t = table[col].unstack(level=1).loc[["Train", "Test"]]
        t = t[["Baseline A", "Band+VWAP", "Band+VWAP Dyn", "SPY buy&hold"]]
        t.plot.bar(ax=ax, rot=0, color=["black", "green", "blue", "red"])
        ax.set_title(title)
        ax.set_xlabel("")
        ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "metrics_comparison.png", dpi=150)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    m = load_market()
    print(f"Loaded {m['n_days']} days: {m['dates'][0].date()} -> {m['dates'][-1].date()} "
          f"(first {LOOKBACK} days are warm-up, no trades)")

    bands = noise_area(m)
    vwap_mark = vwap_at_marks(m)
    lev = leverage_by_day(m)

    live = lev[LOOKBACK:]
    print(f"Leverage (dynamic version): average {live.mean():.2f}x | max {live.max():.2f}x | "
          f"below 1x on {np.mean(live < 1):.0%} of days | at the {MAX_LEVERAGE:.0f}x cap on {np.mean(live >= MAX_LEVERAGE):.0%}")

    # signals are generated once per rule set; the dynamic version reuses the Band+VWAP trades
    trades_a = generate_trades(m, bands, vwap_mark, "A")
    trades_v = generate_trades(m, bands, vwap_mark, "band_vwap")
    runs = {
        "Baseline A": (trades_a, run_account(m, trades_a)),
        "Band+VWAP": (trades_v, run_account(m, trades_v)),
        "Band+VWAP Dyn": (trades_v, run_account(m, trades_v, leverage=lev)),
    }

    first_day = next(iter(runs.values()))[1].index
    spy = pd.Series(m["C"][:, -1], index=m["dates"]).pct_change().loc[first_day]   # close-to-close

    table = print_report(runs, spy)
    cost_sensitivity(m, runs, lev)

    save_plots(runs, spy, table)
    table.to_csv(RESULTS_DIR / "summary.csv")
    for name, (trades, daily) in runs.items():
        tag = name.replace("+", "_").replace(" ", "_")
        trades.to_csv(RESULTS_DIR / f"trades_{tag}.csv", index=False)
        daily.to_csv(RESULTS_DIR / f"daily_{tag}.csv")
    print(f"\nPlots and CSVs saved in {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()