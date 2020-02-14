import silence_tensorflow
import tensorflow.compat.v1 as tf
import tensorflow.contrib.slim as slim
from model import LstmRNN
from config import Config
from utils import (
    validate_tf_1_14_plus,
    get_tensorflow_config
)

if not validate_tf_1_14_plus(tf.version.VERSION):
    raise ImportError('Tensorflow 1.14+ is only supported')


class Trainer:

    def __init__(self, config_file):
        # Get the user configs
        conf = Config(config_file)

        # Get the DAG config
        gconf = get_tensorflow_config()

        # Create the model in the graph
        with tf.Session(config=gconf) as sess:
            self.model = LstmRNN(
                name=conf.name,
                sess=sess,
                cell_dim=conf.cell_dim,
                layers=conf.layers['count'],
                dropout_rate=conf.layers['dropout_rate'],
                tensorboard=conf.tensorboard,
                lstm_size=conf.lstm['size']
            )

            self._list_all_trainables()

    def _list_all_trainables(self):
        """
        List all the trainable variables
        in the model
        :return:
        """
        vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(vars, print_info=True)

    def train(self):
        """
        Train the model
        :return:
        """
        pass

if __name__ == '__main__':
    trainer = Trainer('config.yaml')