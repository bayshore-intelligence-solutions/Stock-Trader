import numpy as np
import pandas as pd
from pathlib import Path
import warnings
from utils import warning_on_one_line
import time


np.random.seed(int(time.time()))


class StockDataset:

    def __init__(self, config, **kwargs):
        # // TODO: Replace the stock_data with data iterator

        datapath = config.data['datapath']
        symbol = config.data['symbol'] + '.csv'

        stock_data = pd.read_csv(Path(datapath).joinpath(symbol))
        cols = config.data['cols']

        self.time_steps = config.lstm['time_step']
        normalize = kwargs.get('normalize', True)
        validation_ratio = config.data['validation_ratio']
        self.X_train, self.X_val, self.y_train, self.y_val = self._prepare_data(
            stock_data, cols, normalize, validation_ratio)
        self.batch_size = config.ops['batch_size']

        # Some assertions
        assert len(self.X_train.shape) == len(self.X_val.shape) == 3
        assert self.X_train.shape[0] == self.y_train.shape[0]
        assert self.X_val.shape[0] == self.y_val.shape[0]
        assert self.time_steps == self.X_train.shape[1] == self.X_val.shape[1]

        # Batch Calculations
        self.num_batches = self.X_train.shape[0] // self.batch_size
        if self.num_batches * self.batch_size < self.X_train.shape[0]:
            self.num_batches += 1

        self.rbi = np.arange(self.num_batches)

    @property
    def size(self, train=True):
        if train:
            return {
                'X': self.X_train.shape,
                'y': self.y_train.shape
            }
        return {
            'X': self.X_val.shape,
            'y': self.y_val.shape
        }

    def _prepare_data(self, data, cols, normalize, validation_ratio):
        # Get the input shape
        input_shape = len(cols)

        # Get the number of cols for each data points
        seq = data[cols].values

        # Normalize the data...
        # Note: This is a time series normalization
        if normalize:
            for i in range(input_shape):
                s = seq[1:, i] / seq[:-1, i] - 1.0
                seq[:, i] = np.append([0.0], s)

        # Split into group of num_steps
        X = np.array([seq[i: i + self.time_steps]
                      for i in range(seq.shape[0] - self.time_steps)
                      ])
        y = np.array([seq[i + self.time_steps]
                      for i in range(seq.shape[0] - self.time_steps)])

        if validation_ratio <= 0.0:
            warnings.formatwarning = warning_on_one_line
            warnings.warn('Using no data for testing!')
            return X, None, y, None
        elif validation_ratio > 0.5:
            raise AttributeError(
                'Using more than 50% of the data for testing'
            )
        else:
            train_size = int(len(X) * (1.0 - validation_ratio))
            train_X, val_X = X[:train_size], X[train_size:]
            train_y, val_y = y[:train_size], y[train_size:]
            return train_X, val_X, train_y, val_y

    @property
    def data(self):
        return {
            'X_train': self.X_train,
            'X_val': self.X_val,
            'y_train': self.y_train,
            'y_val': self.y_val
        }

    def generate_for_one_epoch(self):
        np.random.shuffle(self.rbi)
        for i in self.rbi:
            batch_X = self.X_train[i * self.batch_size: (i + 1) * self.batch_size]
            batch_y = self.y_train[i * self.batch_size: (i + 1) * self.batch_size]

            # Assert all the batches have same number of `time_steps`
            assert set(map(len, batch_X)) == {self.time_steps}
            yield batch_X, batch_y


# if __name__ == '__main__':
#     from config import Config
#     conf = Config('config.yaml')
#
#     # master_card = pd.read_csv(DATA.joinpath('MA.csv'))
#     dataset = StockDataset(config=conf)
#     data = dataset.generate_for_one_epoch()
#     for batch_X, batch_y in data:
#         print(batch_X.shape)
#         print(batch_y.shape)
#         break
