import os
import pandas as pd

def calculate_Amihud_Illiquidity_20d_Median():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'daily_pv.h5')
    result_path = os.path.join(base_dir, 'result.h5')

    df = pd.read_hdf(data_path, key='data')
    df = df.sort_index()

    close = df['$close']
    volume = df['$volume']

    prev_close = close.groupby(level='instrument').shift(1)
    ret = close / prev_close - 1.0

    illiq = ret.abs() / (close * volume)
    illiq = illiq.replace([float('inf'), float('-inf')], float('nan'))

    def winsorize(x):
        lower = x.quantile(0.01)
        upper = x.quantile(0.99)
        return x.clip(lower=lower, upper=upper)

    illiq_winsorized = illiq.groupby(level='datetime').transform(winsorize)

    factor = illiq_winsorized.groupby(level='instrument').transform(lambda x: x.rolling(20).median())
    factor = factor.rename('Amihud_Illiquidity_20d_Median')

    factor.to_frame().to_hdf(result_path, key='data')

if __name__ == '__main__':
    calculate_Amihud_Illiquidity_20d_Median()
