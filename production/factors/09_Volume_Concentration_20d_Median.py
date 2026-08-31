import pandas as pd
import numpy as np

def calculate_Volume_Concentration_20d_Median():
    df = pd.read_hdf('daily_pv.h5', key='data')
    df = df.sort_index()
    volume = df['$volume'].astype('float64')

    # rolling 20-day sums per instrument
    sum_vol = volume.groupby(level='instrument').transform(
        lambda x: x.rolling(20, min_periods=20).sum()
    )
    sum_vol_sq = (volume ** 2).groupby(level='instrument').transform(
        lambda x: x.rolling(20, min_periods=20).sum()
    )

    # daily Herfindahl index over 20-day volume shares:
    # H_{i,d} = sum(volume^2 over 20d) / (sum(volume over 20d))^2
    hhi = sum_vol_sq / (sum_vol ** 2)
    hhi.name = 'daily_herfindahl'

    # cross-sectional winsorization at 1%/99% per trading day
    lower = hhi.groupby(level='datetime').transform(lambda x: x.quantile(0.01))
    upper = hhi.groupby(level='datetime').transform(lambda x: x.quantile(0.99))
    hhi_win = hhi.clip(lower=lower, upper=upper)

    # median of winsorized daily Herfindahl over the last 20 trading days per instrument
    factor = hhi_win.groupby(level='instrument').transform(
        lambda x: x.rolling(20, min_periods=20).median()
    )
    factor.name = 'Volume_Concentration_20d_Median'

    result = factor.to_frame()
    result.to_hdf('result.h5', key='data')


if __name__ == '__main__':
    calculate_Volume_Concentration_20d_Median()
