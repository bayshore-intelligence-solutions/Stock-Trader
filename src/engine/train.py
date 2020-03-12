import silence_tensorflow
import tensorflow.compat.v1 as tf
import tensorflow.contrib.slim as slim
from model import LstmRNN
from config import Config
from utils import (
    validate_tf_1_14_plus,
    get_tensorflow_config
)
from pathlib import Path
import time
from data import StockDataset
from progress.bar import PixelBar

if not validate_tf_1_14_plus(tf.version.VERSION):
    raise ImportError('TensorFlow 1.14+ is only supported')

BASE = Path('./../../').absolute()
DATA = BASE.joinpath('data')


class Train:
    VALID_LOSS = None

    def __init__(self, dataset, config_file):
        # Get the user configs
        self.conf = Config(config_file)
        self._saver = None

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
                lstm_size=self.conf.lstm['size'],
                device=self.conf.ops['device'],
                batch_size=conf.ops['batch_size']
            )

            # Print all the trainable variables
            Train._list_all_trainables()

            # Define Loss, Optimizer and Learning Rate

            # pred and targets are not the actual predictions or labels
            # but the location where during DAG computation, they would
            # be accumulated.

            with tf.name_scope('squared_loss'):
                self.train_loss = Train.squared_loss(self.model.pred,
                                                     self.model.targets,
                                                     name="train_loss")
                self.val_loss = Train.squared_loss(self.model.pred,
                                                   self.model.targets,
                                                   name='val_loss')

            with tf.name_scope('train'):
                self.optim = Train._get_optimizer(self.conf.ops['optimizer'],
                                                  self.conf.ops['learning_rate'])
                self.optim = self.optim.minimize(self.train_loss, name='optim_loss')
                self.train(dataset)

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

    @property
    def saver(self):
        if self._saver is None:
            self._saver = tf.train.Saver(max_to_keep=1)  # Because we rename every checkpoint
        return self._saver

    def save_model(self, path: Path, global_step: int,
                   val_loss: float, epoch: int,
                   tol: float, startTime: str) -> bool:
        """
        This saves the model checkpoint by automatically checking
        if the model has improves from the previous best by a tolerance amount
        given by ```tol```.

        :param path: Path to save the model
        :param global_step: Global step counter of the training
        :param val_loss: Current Validation Loss
        :param epoch: Current Epoch
        :param tol: Tolerance level/Minimum performance improvement from the previous
        :param startTime: Start time since epoch
        :return: True if the model is saved else False
        """
        if not path.joinpath('checkpoints').is_dir():
            raise IOError('Directoy does not exist')

        PATH = path.joinpath("checkpoints").joinpath(startTime)
        NAME = f"Epoch_{epoch}_ValLoss_{val_loss}"

        if Train.VALID_LOSS is None:
            prev_val_loss = -1.0
        else:
            prev_val_loss = Train.VALID_LOSS

        if prev_val_loss != -1.0 and prev_val_loss - tol < val_loss:
            # Validation loss has not improved
            return False

        ckpt_file = PATH.joinpath(NAME)
        self.saver.save(sess=self.model.sess, save_path=ckpt_file,
                        global_step=global_step)
        Train.val_loss = val_loss
        return True

    def train(self, dataset: StockDataset):
        """
        Train the model
        :param dataset: The sequential dataset
        :return:
        """

        # Initialize all the DAG variables
        tf.global_variables_initializer().run()
        start_time = str(int(time.time()))
        global_step = 0
        EPOCHS = self.conf.ops['epochs']
        NUM_BATCHES = dataset.num_batches

        # Write the graph summary in tensorboard
        with tf.summary.FileWriter("./LOGDIR") as gs:
            gs.add_graph(self.model.sess.graph)

        # Training loop
        for epoch in range(EPOCHS):
            epoch_step = 0
            # Returns an iterator only for training data
            data = dataset.generate_for_one_epoch()
            total_training_loss = 0.0
            self.model.training = True
            with PixelBar(f'Epoch {epoch + 1}: ',
                          max=NUM_BATCHES) as bar:
                bar.check_tty = False
                for batch_X, batch_y in data:
                    global_step += 1
                    epoch_step += 1

                    # Training feed dict
                    train_data_feed = {
                        self.model.inputs: batch_X,
                        self.model.targets: batch_y,
                        self.model.keep_prob: 1.0 - conf.layers['dropout_rate']
                    }

                    train_loss = self.model.sess.run(
                        [self.train_loss, self.optim],
                        train_data_feed
                    )
                    # bar.set_postfix(train_loss=round(train_loss[0], 10))
                    bar.suffix = 'Total training Loss: {:.7e}'.format(total_training_loss)
                    bar.next()
                    total_training_loss += train_loss[0]

                # Check the performance on the validation dataset
                val_data_feed = {
                    self.model.inputs: dataset.X_val,
                    self.model.targets: dataset.y_val,
                    self.model.keep_prob: 1.0
                }
                # self.model.training = False  # For dropouts
                val_loss, val_pred = self.model.sess.run(
                    [self.val_loss, self.model.pred],
                    feed_dict=val_data_feed
                )

                print(f'\n\nEpoch: {epoch + 1}, Training Loss: {total_training_loss / NUM_BATCHES}')
                print(f'Epoch: {epoch + 1}, Validation Loss: {val_loss}\n')
                if not self.save_model(conf.root, global_step, val_loss, epoch+1, 0.00001, start_time):
                    print(f'Validation loss has not improved from the previous value {Train.VALID_LOSS}')


if __name__ == '__main__':
    conf = Config('config.yaml')
    dataset = StockDataset(config=conf)
    trainer = Train(dataset, 'config.yaml')
