import os
import sys
import random
import dask.dataframe as dd
from glob import glob
from dotenv import load_dotenv

'''
Load environment variables
'''
load_dotenv()
sys.path.append(os.getenv("SRC_DIR"))

'''
Set up logger
'''
from utils.logger import get_logger
_logs = get_logger(__name__)

'''
Load data
'''
PRICE_DATA = os.getenv("PRICE_DATA")
parquet_files = glob(os.path.join(PRICE_DATA, "**/*.parquet"), recursive=True)
_logs.info(f"Found {len(parquet_files)} parquet files.")

dd_px = dd.read_parquet(parquet_files)
_logs.info("Parquet files loaded into Dask DataFrame.")

'''
Lag features
'''
dd_lags = dd_px.groupby("ticker", group_keys=False).apply(
    lambda x: x.assign(
        Close_lag_1 = x["Close"].shift(1),
        Adj_Close_lag_1 = x["Adj Close"].shift(1)
    )
)
_logs.info("Lag features calculated.")

'''
Returns
'''
dd_rets = dd_lags.assign(
    returns = lambda x: (x["Close"] / x["Close_lag_1"]) - 1
)
_logs.info("Returns calculated.")

'''
High-Low range
'''
dd_feat = dd_rets.assign(
    hi_lo_range = lambda x: x["High"] - x["Low"]
)
_logs.info("High-Low range feature added.")

'''
Compute Dask DataFrame
'''
dd_feat = dd_feat.compute()
_logs.info("Dask DataFrame computed to Pandas.")

'''
Add moving average
'''
random.seed(42)
dd_feat["returns_moving_avg_10"] = (
    dd_feat.groupby("ticker")["returns"]
    .transform(lambda x: x.rolling(10).mean())
)

_logs.info("Moving average added.")

'''
Preview
'''
_logs.info("Previewing final feature set:")
print(dd_feat[["ticker", "Date", "Year", "Close_lag_1", "Adj_Close_lag_1", "returns", "hi_lo_range", "returns_moving_avg_10"]].head(15))

