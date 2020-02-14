from typing import Tuple
import silence_tensorflow
import tensorflow.compat.v1 as tf
from tensorflow.contrib import rnn
from utils import (
    validate_tf_1_14_plus,
    get_tensorflow_config
)
from config import Config

if not validate_tf_1_14_plus(tf.version.VERSION):
    raise ImportError('Tensorflow 1.14+ is only supported')


class LstmRNN:

    def __init__(self,
                 name: str,
                 sess: tf.Session,
                 cell_dim: Tuple[int, int],
                 layers: int,
                 dropout_rate: float,
                 lstm_size: int,
                 tensorboard: bool,
                 **kwargs):
        """
        Creates an LSTM network with the following
        parameters.

        :param name: Name of the LSTM architecture
        :param sess: Tensorflow Session variable
        :param cell_dim: time_steps x input_size
        :param layers: Number of LSTM layers
        :param dropout_rate: 1 - keep_prob rate for LSTM cell
        :param lstm_size: Internal state size inside LSTM cells
        :param tensorboard: Use tensorboard or not
        """

        self.name = name
        self.sess = sess
        self.time_steps, self.input_size = cell_dim
        self.layers = layers
        self.keep_prob = 1 - dropout_rate
        self.lstm_size = lstm_size
        if tensorboard:
            # If using tensorboard
            self.logdir = kwargs.get('logdir')
        else:
            self.logdir = None

        self._build_graph()

    def __str__(self):
        return f'LSTM network of type {self.name} ' \
               f'at {hex(id(self))} of ' \
               f'type {self.__class__.__name__} running on ' \
               f'a Tensorflow session at {hex(id(self.sess))}'

    def _one_rnn_cell(self):
        """
        This create one RNN cell
        :return:
        """
        lstm_cell = rnn.LSTMCell(self.lstm_size,
                                 state_is_tuple=True)
        lstm_cell = rnn.DropoutWrapper(lstm_cell,
                                       output_keep_prob=self.keep_prob)

        return lstm_cell

    def _get_all_placeholders(self):
        self.learning_rate = tf.placeholder(tf.float32, None,
                                            name="learning_rate")
        self.inputs = tf.placeholder(tf.float32,
                                     [None, self.time_steps, self.input_size],
                                     name="inputs")
        self.targets = tf.placeholder(tf.float32, [None, self.input_size],
                                      name="targets")

    @staticmethod
    def _variable_on_device(name, shape, initializer, device='cuda', _id=0):
        """
        Declare a variable on device of `shape` and initialize
        those with the `initializer`.
        """
        if device == 'cuda':
            d = f'/gpu:{_id}'
        else:
            d = '/cpu:0'
        with tf.device(d):
            var = tf.get_variable(name, shape, initializer=initializer)
        return var

    def _build_graph(self):
        # First get all the placeholders
        self._get_all_placeholders()

        # Second, create multiple LSTM cells for all the layers
        cells = rnn.MultiRNNCell(
            [self._one_rnn_cell() for _ in range(self.layers)],
            state_is_tuple=True
        ) if self.layers > 1 else self._one_rnn_cell()

        # Wrap the all the cells into an RNN
        # We shall use dynamic_rnn so that
        """
          1. All the cells for each time_steps in the graph 
            is not created while creating the RNN

          2. We can pass more/less than the sequence length.
        """
        out, _ = tf.nn.dynamic_rnn(cells, self.inputs,
                                   dtype=tf.float32, scope="dynamic_rnn")
        # It returns the output of all the time_steps. However we need the
        # output only from the second last time step. Second last because the
        # output for the last timestep is [0.0, 0.0, 0.0 .... 0.0]. We need
        # the last non-zero output

        # For this we need to get the num_of_time_steps from the output and
        # subtract 1 from it. We could have also wriiten self.time_steps - 1
        # but that would not have been generic.

        # Actual output => (batch_size, num_steps, lstm_size)
        # This converts => (num_steps, batch_size, lstm_size)
        # JUST LIKE THAT!!
        out = tf.transpose(out, [1, 0, 2])

        num_time_steps = int(out.get_shape()[0])
        last_state = tf.gather(out, num_time_steps - 1, name="last_lstm_state")

        # Now that we've got the last_state, we can add it to a linear layer for Regression.
        W = LstmRNN._variable_on_device(
            name='w',
            shape=(self.lstm_size, self.input_size),
            initializer=tf.truncated_normal_initializer(),
        )
        bias = LstmRNN._variable_on_device(
            name='bias',
            shape=(self.input_size,),
            initializer=tf.constant_initializer(0.1),
        )

        self.pred = tf.matmul(last_state, W) + bias
        print(self.pred)


if __name__ == '__main__':
    # Get the architecture config
    conf = Config('config.yaml')

    # Get the DAG config
    gconf = get_tensorflow_config()

    with tf.Session(config=gconf) as sess:
        model = LstmRNN(
            name=conf.name,
            sess=sess,
            cell_dim=conf.cell_dim,
            layers=conf.layers['count'],
            dropout_rate=conf.layers['dropout_rate'],
            tensorboard=conf.tensorboard,
            lstm_size=conf.lstm['size']
        )