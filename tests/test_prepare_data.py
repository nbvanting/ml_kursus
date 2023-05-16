import click
import pandas as pd

from src.prepare_data import TSDataset


@click.command()
@click.option('--length', default=10, type=click.INT, help='length of series')
@click.option('--lags', default=3, type=click.INT, help='Number of steps in the past')
@click.option('--horizon', default=2, type=click.INT, help='Number of steps in the future')
@click.option('--target', default='value_t', type=click.STRING, help='Name of target column')
def main(length: int, lags: int = 4, horizon: int = 4, target: str = 'value_t') -> None:
    ts_data = pd.DataFrame([x for x in range(1, length+1)], columns=[target])
    ts_dataset = TSDataset(ts_data, target, drop_nan=True)
    X, y = ts_dataset.to_supervised(n_lags=lags, horizon=horizon)
    print('Timeseries data:')
    print(ts_data)
    print('\nInput data: X')
    print(X)
    print('\nLabel data: y')
    print(y)

if __name__ == '__main__':
    main()
