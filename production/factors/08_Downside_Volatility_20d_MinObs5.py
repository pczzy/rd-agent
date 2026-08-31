import os
import pandas as pd
import numpy as np

def calculate_Downside_Volatility_20d_MinObs5():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "daily_pv.h5")
    result_path = os.path.join(base_dir, "result.h5")

    df = pd.read_hdf(data_path, key="data")
    df = df.sort_index()

    close = df["$close"]
    # Daily returns per instrument
    ret = close / close.groupby(level="instrument").shift(1) - 1.0

    # Cross-sectional winsorization at 1%/99% per trading day
    win_ret = ret.groupby(level="datetime").transform(
        lambda x: x.clip(x.quantile(0.01), x.quantile(0.99))
    )

    def downside_std(arr):
        neg = arr[arr < 0]
        if len(neg) >= 5:
            return np.std(neg, ddof=1)
        else:
            return np.nan

    # Rolling 20-day downside volatility per instrument, requiring at least 20 observations
    downside_vol = win_ret.groupby(level="instrument").transform(
        lambda x: x.rolling(20, min_periods=20).apply(downside_std, raw=True)
    )

    result = downside_vol.rename("Downside_Volatility_20d_MinObs5")
    result = result.replace([float("inf"), float("-inf")], float("nan"))

    result.to_frame().to_hdf(result_path, key="data")

if __name__ == "__main__":
    calculate_Downside_Volatility_20d_MinObs5()
