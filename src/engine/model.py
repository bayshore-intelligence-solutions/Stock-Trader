import tensorflow as tf



from typing import Tuple
import silence_tensorflow

# from utils import (
#     validate_tf_1_14_plus,
#     get_tensorflow_config
# )

from config import Config

# if not validate_tf_1_14_plus(tf.version.VERSION):
# 	raise ImportError('Tensorflow 1.14+ is only supported')

class LSTM_RNN(tf.keras.Model):

	def __init__(
		self,
		name: str,
		cell_dim: Tuple[int,int],
		layers: int,
		dropout_rate: float,
		lstm_size: int,
		tensorboard: bool,
		**kwargs):

		"""
		Creates an LSTM network with the following
		parameters.

		:param name: Name of the LSTM architecture
		:param cell_dim: time_steps x input_size
		:param layers: Number of LSTM layers
		:param dropout_rate: 1 - keep_prob rate for LSTM cell
		:param lstm_size: Internal state size inside LSTM cells
		:param tensorboard: Use tensorboard or not
		"""
		super(LSTM_RNN, self).__init__()

		self.name_model = name
		self.time_steps, self.input_size = cell_dim
		# print(self.time_steps)
		# print(self.input_size)
		self.n_layers = layers
		#self.keep_prob = 1 - dropout_rate
		self.keep_prob = dropout_rate
		self.lstm_size = lstm_size
		self.device = kwargs.get("device", "cuda")

		if tensorboard:
			self.logdir = kwargs.get("logdir")
		else:
			self.logdir = None

		#The architecture of the body
		self.arch = tf.keras.layers.StackedRNNCells(
            [self._one_rnn_cell(l + 1) for l in range(self.n_layers)],
            state_is_tuple=True) if self.n_layers > 1 else self._one_lstm_cell(1)

		self.output_layer = tf.keras.layers.Dense(1, use_bias=True,
			kernel_initializer = tf.keras.initializers.TruncatedNormal())

	def __str__(self):
		return f'LSTM network of type {self.name_model} '

	def _one_lstm_cell(self, layer):
		"""
		This create one RNN cell
		:param layer: Mention the layer number
		:return: LSTM cell with or without dropout
		"""
		lstm_cell = tf.keras.layers.LSTM(self.lstm_size,
			dropout = self.keep_prob, stateful=True,
			name=f'LSTM_layer_{layer}')

		return lstm_cell

	def call(self, inputs, training=False):

		# x = self.input_layer(inputs)
		x = self.arch(inputs)

		x = self.output_layer(x)

		return(x)


if __name__ == '__main__':
	# Get the architecture config
	conf = Config('config.yaml')

	model = LSTM_RNN(
		name=conf.name,
		cell_dim=conf.cell_dim,
		layers=conf.layers['count'],
		dropout_rate=conf.layers['dropout_rate'],
		tensorboard=conf.tensorboard,
		lstm_size=conf.lstm['size'],
		batch_size=conf.ops['batch_size']
	)
	# model.compile()
	# model.summary()
