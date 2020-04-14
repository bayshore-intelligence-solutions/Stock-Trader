import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
from utils import warning_on_one_line
import time
from pathlib import Path
# import silence_tensorflow


def seed_everything(seed : int) -> None:
	"""
	To seed everything
	"""
	np.random.seed(seed)
	tf.random.set_seed(seed)

class StockDataset:
	"""
	Class to load and process the dataset
	"""
	seed_everything(int(time.time()))

	def __init__(self, config, **kwargs):
		"""
		Fetching all the config and params
		"""
		datapath = config.data['datapath']
		symbol = config.data['symbol'] + '.csv'
		cols = config.data['cols']
		self.nCols = len(cols)
		self.time_steps = config.lstm['time_step']
		self.gap = config.data['gap']
		normalize = kwargs.get('normalize', True)
		validation_ratio = config.data['validation_ratio']
		self.batch_size = config.ops['batch_size']

		#Loading the data
		stock_data = pd.read_csv(Path(datapath).joinpath(symbol))

		#Preparing the data
		self.num_batches, self.train, self.val = self._prepare_data(
			stock_data, cols, normalize, validation_ratio)
		# print(self.val)
		#Assertions
		# assert len(self.X_train.shape) == len(self.X_val.shape) == 3
		# assert self.X_train.shape[0] == self.y_train.shape[0], "X_train must be equal to y_train"
		# assert self.X_val.shape[0] == self.y_val.shape[0], "X_val must be equal to y_val"
		# assert self.time_steps == self.X_train.shape[1] == self.X_val.shape[1]
		# assert self.X_train.shape[0] > self.batch_size, "Batch size greater than training data"


	@staticmethod
	def _normalize(seq: np.ndarray, input_shape: int):
		"""
		Function to normalize the data
		"""
		for i in range(input_shape):
			s = seq[1:, i] / seq[:-1, i] - 1.0
			seq[:, i] = np.append([0.0], s)
		return seq

	def _prepare_data(self, data, cols, normalize, validation_ratio):
		#Get the input shape
		input_shape = len(cols)

		# Get the number of cols for each data points
		seq = data[cols].values

		#Normalizing the data w/ timeseries normalization
		if normalize:
			seq = StockDataset._normalize(seq, input_shape)

		# Suppose train dataset = X which is of len x
		# Then len(y)==len(X); Suppose the sliding window is s
		# if we take groups of X where x_batch = [X[i:time_steps] for i in range((len(X)-s) - time_steps)], we'll get
		# the desired batch numbers omitting the last s from X
		# Now for y, to account for the gap, we'll take
		# y_batch = [Y[i+time_steps+gap] for i in range((len(X)-s) - time_steps)]

		# print(seq.shape)
		# Split into group of num_steps
		X = np.array([seq[i: i + self.time_steps] for i in range((seq.shape[0]-self.gap) - self.time_steps)])  #Taking into account the gap during training
		y = np.array([seq[i + self.time_steps+self.gap] for i in range((seq.shape[0]-self.gap) - self.time_steps)])

		#Validating gap
		assert len(X)==len(y)
		valid_batches = []
		for i in range(self.gap,len(X)):
			if y[i] in X[i-self.gap]:
				# print(np.where(X[i-self.gap]==y[i]))
				valid_batches.append(True)
		if all(valid_batches):
			print("The gap is maintained")
		    # exit()
		else:
			print("The gap is not maintained for all cases")
		    # exit()
		# for i in range(1,len(X)):
		# 	if X[i][self.gap]==y[i-1][0]:
		# 		print("True")
		# 		valid_batches.append(True)

		#Checking validity
		assert all(valid_batches)==True

		if validation_ratio <= 0.0:
			warnings.formatwarning = warning_on_one_line
			warnings.warn('Using no data for testing!')
			#Returns tf data using Dataset api
			n_batches= X.shape[0]//self.batch_size
			X = tf.data.Dataset.from_tensor_slices(X)
			y = tf.data.Dataset.from_tensor_slices(y)
			train = tf.data.Dataset.zip((X,y))
			# print(train.as_numpy_iterator().shape)
			val = tf.data.Dataset.from_tensor_slices((None, None))
			return n_batches, train, val
		elif validation_ratio > 0.5:
			raise AttributeError("Using more than 50% of the data for testing")
		else:
			train_size = int(len(X) * (1.0 - validation_ratio))
			train_X, val_X = X[:train_size], X[train_size:]
			train_y, val_y = y[:train_size], y[train_size:]

			n_batches = train_X.shape[0]//self.batch_size
			#Converting the datastream into tf.data
			X_train = tf.data.Dataset.from_tensor_slices(train_X)
			y_train = tf.data.Dataset.from_tensor_slices(train_y)
			train = tf.data.Dataset.zip((X_train,y_train))
			# print(train.as_numpy_iterator().shape)

			X_val = tf.data.Dataset.from_tensor_slices(val_X)
			y_val = tf.data.Dataset.from_tensor_slices(val_y)
			val = tf.data.Dataset.zip((X_val,y_val))
			# print(val.as_numpy_iterator().shape)
			return n_batches, train, val

	def generate_train_for_one_epoch(self):
		"""
		Generating data for one epoch after shuffling, batching and prfetching one batch
		"""
		self.train_temp = self.train.shuffle(buffer_size = self.num_batches, reshuffle_each_iteration=True)
		#Splitting in batches
		self.train_temp = self.train_temp.batch(self.batch_size, drop_remainder=True)
		#Making iterator
		# iterator = self.train.make_initializable_iterator()
		#Prefetching one batch at runtime
		for (x_temp, y_temp) in self.train_temp:
			# Assert all the batches have same number of `time_steps`
			# assert set(map(len, x_temp)) == {self.time_steps}
			# print(x_temp.shape)
			# print(y_temp.shape)
			yield x_temp, y_temp


	def generate_val_for_one_epoch(self):
		"""
		Generating data for one epoch after shuffling, batching and prfetching one batch
		"""
		self.val_temp = self.val.shuffle(buffer_size=self.num_batches, reshuffle_each_iteration=False)
		#Splitting in batches
		self.val_temp = self.val_temp.batch(self.batch_size, drop_remainder=True)
		#Prefetching one batch at runtime
		for (x_temp, y_temp) in self.val_temp:
			# Assert all the batches have same number of `time_steps`
			# assert set(map(len, x_temp)) == {self.time_steps}
			yield x_temp, y_temp

	@staticmethod
	def prepare_for_test_single(data: pd.DataFrame, **kwargs):
		"""
		Used to prepare the data for the testing purpose
		:param data: pd.DataFrame
		:return:
		"""
		nSeq, nCols = data.shape

		# # Ensure the sequence length is correct
		# assert nSeq == self.time_steps
		# # Ensure the input size is correct
		# assert nCols == self.nCols

		normalize = kwargs.get('normalize', True)
		seq = data.values
		if normalize:
		    seq = StockDataset._normalize(seq, nCols)
		# x_test, y_test = np.expand_dims(seq[0:-1], axis=0), np.expand_dims(seq[-1], axis=0)
		# x_test, y_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
		# return x_test
		return np.expand_dims(seq[0:-1], axis=0), np.expand_dims(seq[-1], axis=0)


# if __name__ == '__main__':
#     import yfinance as yf
#     data = yf.download('V', period="2d", interval="1m", threads=1)
#     df = data.tail(130)[["Open", "Close"]]
#     X, y = StockDataset(df)
#     print(X.shape)
#     print(y.shape)
