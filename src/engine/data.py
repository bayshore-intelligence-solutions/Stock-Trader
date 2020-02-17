import numpy as np
import pandas as pd
from pathlib import Path
import warnings
from utils import warning_on_one_line
from tqdm import tqdm


np.random.seed(42)
BASE = Path('./../../').absolute()
DATA = BASE.joinpath('data')


class StockDataset:

    def __init__(self, stock_data, num_steps, cols, **kwargs):
        # // TODO: Replace the stock_data with data iterator
        self.time_steps = num_steps
        normalize = kwargs.get('normalize', True)
        test_ratio = kwargs.get('test_ratio', 0.2)
        self.X_train, self.X_test, self.y_train, self.y_test = self._prepare_data(
            stock_data, cols, normalize, test_ratio)
        self.batch_size = kwargs.get('batch_size', 16)

        # Some assertions
        assert len(self.X_train.shape) == len(self.X_test.shape) == 3
        assert self.X_train.shape[0] == self.y_train.shape[0]
        assert self.X_test.shape[0] == self.y_test.shape[0]
        assert self.time_steps == self.X_train.shape[1] == self.X_test.shape[1]

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
            'X': self.X_test.shape,
            'y': self.y_test.shape
        }

    def _prepare_data(self, data, cols, normalize, test_ratio):
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

        if test_ratio <= 0.0:
            warnings.formatwarning = warning_on_one_line
            warnings.warn('Using no data for testing!')
            return X, None, y, None
        elif test_ratio > 0.5:
            raise AttributeError(
                'Using more than 50% of the data for testing'
            )
        else:
            train_size = int(len(X) * (1.0 - test_ratio))
            train_X, test_X = X[:train_size], X[train_size:]
            train_y, test_y = y[:train_size], y[train_size:]
            return train_X, test_X, train_y, test_y

    @property
    def data(self):
        return {
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test
        }

    def generate_for_one_epoch(self):
        np.random.shuffle(self.rbi)
        for i in self.rbi:
            batch_X = self.X_train[i * self.batch_size: (i + 1) * self.batch_size]
            batch_y = self.y_train[i * self.batch_size: (i + 1) * self.batch_size]

            # Assert all the batches have same number of `time_steps`
            assert set(map(len, batch_X)) == {self.time_steps}
            yield batch_X, batch_y


if __name__ == '__main__':
    visa = pd.read_csv(DATA.joinpath('visa.csv'))
    # master_card = pd.read_csv(DATA.joinpath('master_card.csv'))
    dataset = StockDataset(stock_data=visa, num_steps=30,
                           cols=['close', 'open'],
                           test_ratio=0.2)
    data = dataset.generate_for_one_epoch()
    print(dataset.size['X'][0] // 16)
    count = 0
    for batch_X, batch_y in tqdm(data, total=2143, ascii=True):
        count += 1
    print(count)
