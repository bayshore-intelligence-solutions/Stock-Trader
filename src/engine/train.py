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
    raise ImportError('TensorFlow 1.14+ is only supported')


class Trainer:

    def __init__(self, config_file):
        # Get the user configs
        self.conf = Config(config_file)

        # Get the DAG config
        gconf = get_tensorflow_config()

        # Create the model in the graph
        with tf.Session(config=gconf) as sess:
            self.model = LstmRNN(
                name=self.conf.name,
                sess=sess,
                cell_dim=self.conf.cell_dim,
                layers=self.conf.layers['count'],
                dropout_rate=self.conf.layers['dropout_rate'],
                tensorboard=self.conf.tensorboard,
                lstm_size=self.conf.lstm['size']
            )

            # Print all the trainable variables
            Trainer._list_all_trainables()

            # Define Loss, Optimizer and Learning Rate

            # pred and targets are not the actual predictions or labels
            # but the location where during DAG computation, they would
            # be accumulated.
            self.train_loss = Trainer.squared_loss(self.model.pred,
                                                   self.model.targets,
                                                   name="train_loss")

            self.test_loss = Trainer.squared_loss(self.model.pred,
                                                  self.model.targets,
                                                  name='test_loss')

            self.optim = Trainer._get_optimizer(self.conf.ops['optimizer'],
                                                self.conf.ops['learning_rate'])

    @staticmethod
    def _get_optimizer(optimizer, eta):
        """
        Get the correct optimizer based on the
        mentioned Learning rate
        :param optimizer:
        :param eta:
        :return:
        """
        if optimizer == 'adam':
            return tf.train.AdamOptimizer(learning_rate=eta, name=optimizer)
        elif optimizer == 'adagrad':
            return tf.train.AdagradOptimizer(learning_rate=eta, name=optimizer)
        elif optimizer == 'rmsprop':
            return tf.train.RMSPropOptimizer(learning_rate=eta, name=optimizer)
        raise NotImplementedError(f'Optimizer {optimizer} is not implemented')

    @staticmethod
    def squared_loss(pred, true, name):
        """
        We are using squared loss. But
        we can define any other loss here
        :param pred: Predicted vector
        :param true: True response
        :param name: Name of the graph operation
        :return:
        """
        return tf.reduce_mean(tf.square(pred - true), name=name)

    @staticmethod
    def _list_all_trainables():
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

        # Initialize all the DAG variables
        tf.global_variables_initializer().run()


        EPOCHS = self.conf.ops['epochs']
        BATCH_SIZE = self.conf.ops['batch_size']




if __name__ == '__main__':
    trainer = Trainer('config.yaml')
