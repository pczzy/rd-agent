import pandas as pd


def calculate_Momentum_10d():
    df = pd.read_hdf("daily_pv.h5", key="data")
    df = df.sort_index()

    close = df["$close"]
    momentum = close / close.groupby(level="instrument").shift(10) - 1
    momentum.name = "Momentum_10d"

    result = momentum.to_frame()
    result.to_hdf("result.h5", key="data")
    return result


if __name__ == "__main__":
    calculate_Momentum_10d()
