import silence_tensorflow
from model import LSTM_RNN
from config import Config
import os
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
from past.builtins import filter
import silence_tensorflow
from data import StockDataset
from train import Train
import yfinance as yf

def _get_live_data():
	data = yf.download('V', period="2d", interval="1m",
					   threads=1, progress=False)
	data = data.tail(31)[["Open", "Close"]]
	return data

def test(conf):
	"""
	For testing the model
	"""
	model = LSTM_RNN(name=conf.name, cell_dim=conf.cell_dim,layers=conf.layers['count'],dropout_rate=conf.layers['dropout_rate'],tensorboard=conf.tensorboard,lstm_size=conf.lstm['size'],device=conf.ops['device'],batch_size=conf.ops['batch_size'])

	data = _get_live_data()
	# Prepare for the model
	X_test, y_test = StockDataset.prepare_for_test_single(data)

	print("Creating the checkpoint manager")
	checkpoint_dir = os.getcwd()+"/checkpoints/" + f"gap_{conf.data['gap']}/"
	ckpt = tf.train.Checkpoint(step=tf.Variable(0), LSTM_RNN=model)
	ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=11)

	path = ckpt_manager.latest_checkpoint
	ckpt.restore(path).expect_partial()

	print("\nModel Restored...")

	pred = model(X_test)
	loss_mad = Train.mad(pred, y_test)
	loss_mse = Train.squared_loss(pred, y_test)

	print("Test loss mad: " + str(loss_mad.numpy()))
	print("Test loss mse: " + str(loss_mse.numpy()))

if __name__ == '__main__':
	conf = Config('config.yaml')
	test(conf)
