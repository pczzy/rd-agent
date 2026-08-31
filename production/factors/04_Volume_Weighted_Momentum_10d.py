import pandas as pd

def calculate_Volume_Weighted_Momentum_10d():
    df = pd.read_hdf("daily_pv.h5", key="data")
    df = df.sort_index()

    close = df["$close"]
    volume = df["$volume"]

    daily_return = close / close.groupby(level="instrument").shift(1) - 1

    data = pd.DataFrame({"daily_return": daily_return, "volume": volume})

    def vw_mom(group):
        numerator = (group["daily_return"] * group["volume"]).rolling(window=10, min_periods=10).sum()
        denominator = group["volume"].rolling(window=10, min_periods=10).sum()
        return numerator / denominator

    factor = data.groupby(level="instrument", group_keys=False).apply(vw_mom)
    factor = factor.astype("float64")
    factor.name = "Volume_Weighted_Momentum_10d"

    result = factor.to_frame()
    result = result.sort_index()
    result.to_hdf("result.h5", key="data")
    return result

if __name__ == "__main__":
    calculate_Volume_Weighted_Momentum_10d()
