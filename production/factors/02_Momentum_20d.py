import os
import pandas as pd


def calculate_Momentum_20d():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "daily_pv.h5")
    result_path = os.path.join(base_dir, "result.h5")

    df = pd.read_hdf(data_path, key="data")
    df = df.sort_index()

    close = df["$close"]
    shifted = close.groupby(level="instrument").shift(20)
    momentum = close / shifted - 1.0

    result = momentum.rename("Momentum_20d")
    result = result.replace([float("inf"), float("-inf")], float("nan"))

    result.to_frame().to_hdf(result_path, key="data")


if __name__ == "__main__":
    calculate_Momentum_20d()
