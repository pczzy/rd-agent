import pandas as pd

def calculate_Intraday_Range_10d():
    df = pd.read_hdf("daily_pv.h5", key="data")
    df = df.sort_index()

    high = df["$high"]
    low = df["$low"]
    close = df["$close"]

    daily_range_ratio = (high - low) / close

    factor = daily_range_ratio.groupby(level="instrument").transform(
        lambda x: x.rolling(window=10, min_periods=10).mean()
    )
    factor.name = "Intraday_Range_10d"

    result = factor.to_frame()
    result.to_hdf("result.h5", key="data")
    return result

if __name__ == "__main__":
    calculate_Intraday_Range_10d()
