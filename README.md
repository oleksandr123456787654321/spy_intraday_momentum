# Intraday Momentum on SPY: replication and my own variation

This project replicates **"Beat the Market: An Effective Intraday Momentum Strategy for S&P500 ETF (SPY)"**
(Zarattini, Aziz & Barbon, Swiss Finance Institute Research Paper 24-97, 2025) and then tests one idea of my own:
a stop rule that switches depending on the VIX.

---

## 1. The paper's strategy

SPY is traded only inside the day. Whatever is open gets closed before the session ends.

1. **Noise Area.** For each time of day, I take the average absolute move from the open over the previous
   14 days (call it sigma). The bands are `open x (1 +/- sigma)`. While price stays inside them, the move
   looks like normal noise, so I do nothing. The bands are built around `max/min(open, previous close)`, so an
   overnight gap pushes one of them further away.
2. **Entry.** I only decide at 10:00, 10:30, ..., 15:30. Price above the upper band means go long, below the
   lower band means go short, inside means stay flat.
3. **Exit.** Everything is closed at the end of the day. I compare two stop rules:
   - **Loose stop (Baseline A):** only exit early if price crosses the *opposite* band, and then flip the position.
   - **Tight stop (Band+VWAP):** exit as soon as price falls back below `max(upper band, VWAP)` for a long, or
     rises above `min(lower band, VWAP)` for a short.
4. **Sizing.** Either 100% of the previous day's equity, or the paper's volatility-targeted version
   (aim for 2% daily volatility, leverage capped at 4x).

Why it might work: markets under-react to news, and options dealers hedging their positions can push intraday
trends further. Where it might fail: quiet, choppy days where breakouts fade, sharp late-day reversals, and
trading costs.

---

## 2. Strategies tested

| Name | Stop rule | Size | File |
|---|---|---|---|
| Baseline A | opposite band | 100% of equity | `main.py` |
| Band+VWAP | current band or VWAP (tight) | 100% of equity | `main.py` (the extension) |
| Band+VWAP Dyn | same as Band+VWAP | 2% volatility target, max 4x | `main.py` (the paper's final model) |
| Switch VIX | loose stop if yesterday's VIX closed above 20, otherwise tight stop | 100% of equity | `switch_vix.py` (my own idea) |
| Switch VIX Dyn | same as Switch VIX | 2% volatility target, max 4x | `switch_vix.py` |
| SPY buy & hold | benchmark | | all |

**The VIX rule.** I wanted a rule with no numbers fitted on my data, so I fixed it up front. The threshold of 20
is the usual line between a normal and an elevated VIX. The direction (loose stop when the VIX is high) comes from
the paper's finding that the strategy works better in high-VIX markets, where trends are bigger and should be
allowed to run. In calm markets I wanted to cut losses early. 
---

## 3. Files

```
data.py              downloads and cleans 1-minute SPY bars (Alpaca) -> spy_1min.csv
main.py              Noise Area, Baseline A, Band+VWAP, dynamic sizing, metrics, plots, slippage table
switch_vix.py        the VIX switch, compared with the three strategies above
cache_1min/          raw download chunks created by data.py (safe to delete)
spy_1min.csv         cleaned minute data created by data.py
vix_daily.csv        daily VIX close, created by switch_vix.py the first time it runs
results/             outputs of main.py
results_switch_vix/  outputs of switch_vix.py
```

---

## 4. Setup

You need Python 3.10 or newer and these packages:

```
pip install numpy pandas matplotlib alpaca-py yfinance
```

You also need Alpaca API keys (a free paper-trading account is enough for historical data). Set them as
environment variables in the terminal you run `data.py` from. Don't put them in the code or upload them anywhere.

```
# Mac / Linux
export ALPACA_KEY="your_key_id"
export ALPACA_SECRET="your_secret_key"

# Windows PowerShell
$env:ALPACA_KEY="your_key_id"
$env:ALPACA_SECRET="your_secret_key"
```

---

## 5. How to run it

From the project folder, in this order:

```
python data.py          # 1. download and clean the data (takes a few minutes the first time)
python main.py          # 2. base strategies, extension, dynamic sizing, report and plots
python switch_vix.py    # 3. the VIX switch (downloads the VIX once and caches it)
```

`data.py` saves every 60-day chunk in `cache_1min/`, so if it stops halfway, just run it again and it carries on.
`switch_vix.py` only needs internet the first time. All the settings (lookback, costs, train/test date, VIX
threshold, leverage) are constants at the top of each file.

---

## 6. Data

- **Source:** Alpaca historical bars, free tier, **IEX feed**, 1-minute bars, unadjusted prices
  (data goes from 2020-07-27 to 2026-10-02)
- **Cleaning rules** (all in `data.py`):
  1. Unadjusted prices, so there are no fake dividend gaps.
  2. Regular hours only (09:30 to 15:59, New York time).
  3. Days that aren't a full session (early closes, for example) are dropped.
  4. Minutes with no trades are filled from the previous close, with volume 0.
  5. Days with more than 10% missing minutes are dropped.
- **Result:** 1,504 usable days, with 1.39% of minutes filled.
- **VIX:** daily close of `^VIX` from Yahoo Finance through `yfinance`. For day *d* I use the close of the last
  trading day *before* *d*, which is known at the open.

The IEX feed is one exchange only. Prices are close to the real market, but volume is just a small slice of it.
That matters for VWAP (see sections 12 and 13).

---

## 7. Costs and sizing

- **Commission:** $0.0035 per share (Interactive Brokers entry-level rate), on every entry and exit.
- **Slippage:** $0.001 per share per execution (the paper's live measurement).
- **Shares** = previous day's equity x leverage / today's open, rounded down. Equity compounds daily.
- `main.py` also prints the Sharpe ratio for slippage from $0 to $0.02 per share.
- Leverage in the "Dyn" versions is free in my backtest: no interest on borrowed money, no margin or
  day-trading rules.

---

## 8. Train / test split

- **Train:** up to and including 2023-12-31. **Test:** from 2024-01-01 on. I fixed the date before looking at any results.
- The strategies run once, continuously, and the daily returns are then split into the two periods. Each period's
  metrics come from its own daily returns.
- I used the paper's parameters as they are (14-day lookback, volatility multiplier 1, 2% target, 4x cap) and
  didn't tune them on my data. The VIX switch has no fitted numbers either.


---

## 9. Metrics

- **Annualized return:** `(final / initial)^(252 / days) - 1`
- **Annualized volatility:** standard deviation of daily returns x sqrt(252)
- **Sharpe ratio:** annualized return / annualized volatility (risk-free rate set to 0, like the paper)
- Also reported: maximum drawdown, hit ratio (winning days among days with a position), skewness, alpha and
  beta against SPY (with the t-statistic of alpha), number of trades, and costs as a share of gross profit.
- SPY buy & hold uses unadjusted prices, so dividends (about 1.3% a year) aren't included.

---

## 10. Outputs

| File | What it shows |
|---|---|
| `results/equity_curves.png` | growth of $1 on a log scale, with the train/test boundary |
| `results/metrics_comparison.png` | Sharpe, annualized return and volatility, train vs test |
| `results/summary.csv`, `trades_*.csv`, `daily_*.csv` | full tables and trade logs |
| `results_switch_vix/equity_curves_switch_vix.png` | the same charts with the VIX switch added |
| `results_switch_vix/metrics_comparison_switch_vix.png` | |
| `results_switch_vix/summary_switch_vix.csv`, `trades_Switch_VIX.csv`, `daily_*.csv` | |

Colors: Baseline A black, Band+VWAP green, Band+VWAP Dyn blue, SPY red, Switch VIX orange, Switch VIX Dyn gold.

---

## 11. Results (2020-08 to 2026-10, after costs)

| | Train Sharpe | Train return / vol | Test Sharpe | Test return / vol |
|---|---|---|---|---|
| Baseline A | 0.51 | 4.8% / 9.4% | 0.35 | 3.0% / 8.6% |
| Band+VWAP | 1.43 | 9.7% / 6.8% | -0.01 | -0.1% / 5.8% |
| Band+VWAP Dyn | 1.55 | 21.6% / 13.9% | 0.00 | 0.1% / 14.8% |
| Switch VIX | 0.56 | 5.0% / 9.0% | 0.27 | 2.2% / 8.1% |
| Switch VIX Dyn | 0.73 | 12.6% / 17.4% | 0.21 | 3.4% / 16.5% |
| SPY buy & hold | 0.62 | 11.0% / 17.9% | 1.31 | 20.2% / 15.4% |

(Returns and volatility are annualized.) What I take from this:

- In the train period the tight VWAP stop reproduces the paper's pattern: higher Sharpe, smaller drawdown,
  positive skew, more trades with a lower win rate. In the test period that improvement is gone.
- Dynamic sizing roughly doubles both risk and return but barely changes the Sharpe ratio, which matches what the
  paper found.
- Every strategy lags SPY buy & hold in the test period, which was a strong market.
- Costs are about 7% to 11% of gross profit, and the conclusions don't depend on the slippage assumption.
- **VIX switch:** in train it gave up most of the tight stop's advantage (Sharpe 1.43 down to 0.56). In test it
  looks better than Band+VWAP, but it used the loose stop on only 19% of test days (123 days), and the
  difference is within statistical noise. So I have no strong evidence that the VIX level tells you which stop to use.

**My main guess for why my results are weaker than the paper's:** the data. I only had the IEX feed, and IEX
volume is a small slice of the real market volume. VWAP is built from volume, so mine is probably not the true VWAP,
and the VWAP stop is the part of the strategy that depends most on it. If the stop sits in the wrong place, that
could explain why its advantage disappeared in the test period. This is a guess, not something I proved. The same
data gave good VWAP results in the train period, so it can't be the whole story. To check it properly I'd need to
rerun everything with consolidated minute data, as the authors used.

---

## 12. Limitations

- **Data:** IEX is a single exchange. The open and close prices can differ slightly from the consolidated tape,
  and volume is only a small share of the market, so my VWAP is approximate. I think this is the most likely reason
  my results differ from the paper's, but I haven't tested it.
- **Short sample:** about 6 years, and it doesn't include 2008 or March 2020. With 2.6 years of test data the
  standard error of a Sharpe ratio is large (roughly 0.6), so many differences can't be told apart from noise.
- **Simplified costs:** flat slippage per share, no interest on leverage, no margin or day-trading rules, and no
  market impact (my simulated account is small).
- **Other possible reasons for the gap with the paper:** the timing convention I used, regime changes, sample
  noise, or the strategy losing its edge since publication. I didn't establish which of these apply.
- **Dropped days:** days removed during cleaning are missing from the series, so the lookbacks span them.

---

## 13. Reference

Zarattini, C., Aziz, A., & Barbon, A. (2025). *Beat the Market: An Effective Intraday Momentum Strategy for
S&P500 ETF (SPY)*. Swiss Finance Institute Research Paper No. 24-97.
