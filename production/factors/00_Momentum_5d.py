import os
import pandas as pd


def calculate_Momentum_5d():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, 'daily_pv.h5')
    output_path = os.path.join(base_dir, 'result.h5')

    data = pd.read_hdf(input_path, key='data')
    close = data['$close'].sort_index()

    shifted = close.groupby(level='instrument', sort=False).shift(5)
    momentum = (close / shifted - 1.0).astype('float64')

    result = momentum.to_frame('Momentum_5d')
    result.to_hdf(output_path, key='data', mode='w')


if __name__ == '__main__':
    calculate_Momentum_5d()
