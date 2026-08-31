import pandas as pd
import numpy as np

def calculate_Return_Volume_Corr_20d():
    df = pd.read_hdf('daily_pv.h5', key='data')
    df = df.sort_index()

    close = df['$close']
    volume = df['$volume']

    ret = close.groupby(level='instrument').pct_change()
    ret.name = 'ret'
    volume.name = 'volume'

    data = pd.concat([ret, volume], axis=1)

    corr = data.groupby(level='instrument', group_keys=False).apply(
        lambda g: g['ret'].rolling(20, min_periods=20).corr(g['volume'])
    )

    lower = corr.groupby(level='datetime').transform(lambda x: x.quantile(0.01))
    upper = corr.groupby(level='datetime').transform(lambda x: x.quantile(0.99))
    factor = corr.clip(lower=lower, upper=upper)

    factor.name = 'Return_Volume_Corr_20d'
    result = factor.to_frame()
    result.to_hdf('result.h5', key='data')

if __name__ == '__main__':
    calculate_Return_Volume_Corr_20d()