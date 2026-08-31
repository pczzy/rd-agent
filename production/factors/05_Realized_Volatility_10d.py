import pandas as pd

def calculate_Realized_Volatility_10d():
    df = pd.read_hdf("daily_pv.h5", key="data")
    df = df.sort_index()

    close = df["$close"]
    ret = close / close.groupby(level="instrument").shift(1) - 1
    vol = ret.groupby(level="instrument").transform(
        lambda x: x.rolling(window=10).std(ddof=1)
    )
    vol.name = "Realized_Volatility_10d"
    vol.index = vol.index.set_names(["datetime", "instrument"])
    result = vol.to_frame()
    result.to_hdf("result.h5", key="data")
    return result

if __name__ == "__main__":
    calculate_Realized_Volatility_10d()
