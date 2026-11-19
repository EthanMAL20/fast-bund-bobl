# Walk-forward validation — v11 (Bund only)

Grid: RSI period [10, 14, 20] x volatility-filter cap [0.04, 0.05, 0.06]. Each test year, the configuration with the best Sharpe ratio over the 3 previous calendar years is frozen and traded for that year. Net of all costs.

| Test year | Calibration | RSI | Vol cap | Out-of-sample return |
|---|---|---:|---:|---:|
| 2018 | 2015-2017 | 20 | 5% | -9.3% |
| 2019 | 2016-2018 | 14 | 5% | +5.9% |
| 2020 | 2017-2019 | 14 | 5% | +29.4% |
| 2021 | 2018-2020 | 14 | 5% | +5.8% |
| 2022 | 2019-2021 | 20 | 5% | +11.2% |
| 2023 | 2020-2022 | 20 | 5% | +4.7% |
| 2024 | 2021-2023 | 10 | 5% | +1.5% |
| 2025 | 2022-2024 | 20 | 4% | -3.0% |
| 2026 | 2023-2025 | 10 | 4% | +10.2% |

| 2018–2026 | Out-of-sample | Fixed v11 settings |
|---|---:|---:|
| Annualised return | 5.9% | 5.2% |
| Volatility | 7.1% | 7.9% |
| Max drawdown | -15.0% | -14.4% |
| Sharpe ratio | 0.68 | 0.53 |

Excluding 2020, the out-of-sample Sharpe ratio is 0.24: the edge is concentrated in stress periods, as is typical of trend signals. The grid is centred on settings already known to work, so the test is somewhat optimistic.

## Why the Bobl leg was removed

Naive rule: long Bobl until 2022-01-07, short afterwards, constant leverage, no filter. Sum of daily returns (not compounded):

| Year | v10 Bobl leg | Naive regime bet |
|---|---:|---:|
| 2015 | +8.3% | +10.1% |
| 2016 | +16.9% | +27.4% |
| 2017 | +12.4% | +0.0% |
| 2018 | +18.5% | +12.7% |
| 2019 | +10.3% | +8.3% |
| 2020 | +7.7% | +5.3% |
| 2021 | -7.8% | -5.7% |
| 2022 | +27.4% | +66.0% |
| 2023 | +16.8% | -18.0% |
| 2024 | +17.2% | +19.9% |
| 2025 | +8.3% | +4.2% |
| 2026 | +12.1% | +16.7% |
| Total | +148.2% | +146.9% |
