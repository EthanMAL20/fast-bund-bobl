# Data

| Folder | Content | In git |
|---|---|---|
| `sample/` | Synthetic random-walk series (`scripts/make_sample_data.py`), same format as the real data | yes |
| `raw/` | Real market data imported from the Excel model (`fastbb import-excel`) | **no** — licensed |

CSV format (one file per series, dates ascending, trading days only):

| File | Columns | Source |
|---|---|---|
| `futures.csv` | `date, bund, bobl` | Bloomberg `RX1 Comdty`, `OE1 Comdty` (PX_SETTLE) |
| `hsi.csv` | `date, value` | Bloomberg `HSI Index` (PX_LAST) |
| `dow_jones.csv` | `date, value` | Bloomberg `INDU Index` (PX_LAST) |
| `slope_2y5y.csv` | `date, value` | `EUSA2 Curncy` − `EUSA5 Curncy`, in % |
| `nfp.csv` | `ref_month, release_date, change` | BLS / FRED `PAYEMS` |
| `ecb_deposit_rate.csv` | `effective_date, rate` | ECB key interest rates |
| `eurostoxx50_tr.csv`, `cac40_tr.csv` | `date, value` | optional benchmarks |
