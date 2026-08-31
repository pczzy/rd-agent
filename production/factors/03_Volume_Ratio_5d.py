import pandas as pd


def calculate_Volume_Ratio_5d():
    df = pd.read_hdf("daily_pv.h5", key="data")
    df = df.sort_index()

    volume = df["$volume"]
    volume_ma5 = volume.groupby(level="instrument").transform(
        lambda x: x.rolling(window=5, min_periods=5).mean()
    )

    factor = volume / volume_ma5 - 1
    factor.name = "Volume_Ratio_5d"

    result = factor.to_frame()
    result.to_hdf("result.h5", key="data")


if __name__ == "__main__":
    calculate_Volume_Ratio_5d()
