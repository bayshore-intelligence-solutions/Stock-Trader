import silence_tensorflow
from model import LSTM_RNN
from config import Config
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
# from tqdm import tqdm

import datetime
from pathlib import Path
import time
from data import StockDataset
from progress.bar import PixelBar
# import os

BASE = Path('./../../').absolute()
DATA = BASE.joinpath('data')

class Train:
	VALID_LOSS = None

	def __init__(self, dataset, model, ckpt, ckpt_manager, config_file):
		#Get the user configs
		self.conf = Config(config_file)
		self._saver = None

		self.model = model
		self.ckpt = ckpt
		self.ckpt_manager = ckpt_manager

		#Initilizing optimizer and loss
		self.optim = Train._get_optimizer(self.conf.ops["optimizer"],
			self.conf.ops["learning_rate"])

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
			return tf.keras.optimizers.Adam(learning_rate=eta, name=optimizer)
		elif optimizer == 'adagrad':
			return tf.keras.optimizers.Adagrad(learning_rate=eta, name=optimizer)
		elif optimizer == 'rmsprop':
			return tf.keras.optimizers.RMSprop(learning_rate=eta, name=optimizer)
		raise NotImplementedError(f'Optimizer {optimizer} is not implemented')

	@staticmethod
	def squared_loss(pred, true):
		"""
		We are using squared loss. But
		we can define any other loss here
		:param pred: Predicted vector
		:param true: True response
		:param name: Name of the graph operation
		:return:
		"""
		return tf.reduce_mean(tf.square(pred - true))


	@staticmethod
	def mad(pred, true):
		"""
		We are using the mean absolute deviation
		"""
		return tf.reduce_mean(tf.abs(pred-true))


	def train_step(self, x_batch, y_batch):
		# train_loss_mad = tf.keras.metrics.Mean("train_loss_mad", dtype=tf.float32)
		# train_loss_mse = tf.keras.metrics.Mean("train_loss_mse", dtype=tf.float32)
		with tf.GradientTape() as tape:
			pred = self.model(x_batch, training=True)
			loss_mad = self.mad(pred, y_batch)
			loss_mse = self.squared_loss(pred, y_batch)
		grads = tape.gradient(loss_mad, self.model.trainable_variables)
		self.optim.apply_gradients(zip(grads, self.model.trainable_variables))

		return (loss_mad, loss_mse)

	def val_step(self, x_val, y_val):
		# val_loss_mad = tf.keras.metrics.Mean("val_loss_mad", dtype=tf.float32)
		# val_loss_mse = tf.keras.metrics.Mean("val_loss_mse", dtype=tf.float32)
		pred = self.model(x_val)
		loss_mad = self.mad(pred, y_val)
		loss_mse = self.squared_loss(pred, y_val)

		return (loss_mad,loss_mse)

	@property
	def saver(self):
		if self._saver is None:
			self._saver = tf.saved_model
		return self._saver

	def save_model(self, path: Path, global_step: int,
		epoch: int, val_loss: float, tol: float, startTime: str) -> bool:

		"""
		This function saves the model checkpoint by comparing improvement
		to the previous epoch with a tolerance value

		:param model: The model to save
		:param path: Path to save the model
		:param global_step: Global step counter of the training
		:param epoch: Current Epoch
		:param val_loss: Current Validation Loss
		:param tol: Tolerance level/Minimum performance improvement from the previous
		:param startTime: Start time since epoch
		:return: True if the model is saved else False
		"""
		if not path.joinpath("checkpoints").is_dir():
			# Makes a new directory in case one is missing
			os.mkdir(path.joinpath("checkpoints"))

		PATH = path.joinpath("checkpoints").joinpath(startTime)
		NAME = f"{epoch}"
		# NAME = f"{epoch}"
		if Train.VALID_LOSS is None:
			prev_val_loss = -1.0
		else:
			prev_val_loss = Train.VALID_LOSS

		if prev_val_loss != -1.0 and prev_val_loss - tol < val_loss:
			# Validation loss has not improved
			return False

		ckpt_manager.save(checkpoint_number=int(ckpt.step))
		print("Saved checkpoint for step {}".format(int(ckpt.step)))
		ckpt.step.assign_add(1)
		Train.VALID_LOSS = val_loss
		return True

	def train(self, dataset: StockDataset):
		"""
		Train the model
		:param dataset: The sequential dataset
		:return:
		"""
		start_time = str(int(time.time()))
		global_step = 0

		EPOCHS = self.conf.ops["epochs"]
		NUM_BATCHES = dataset.num_batches

		# Write the graph summary in tensorboard
		current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		# train_log_dir = os.getcwd() + '/logs/gradient_tape/' + current_time + '/train'
		# val_log_dir = os.getcwd() + '/logs/gradient_tape/' + current_time + '/val'
		train_log_dir = os.getcwd() + '/logs/gradient_tape/' + '/train'
		val_log_dir = os.getcwd() + '/logs/gradient_tape/' + '/val'
		train_summary_writer = tf.summary.create_file_writer(train_log_dir)
		val_summary_writer = tf.summary.create_file_writer(val_log_dir)
		############### Tensorboard not implemented #################
		try:
			for epoch in range(EPOCHS):
				epoch_step = 0
				total_train_loss_mad = 0.0
				total_train_loss_mse = 0.0

				total_val_loss_mad = 0.0
				total_val_loss_mse = 0.0
				# Returns an iterator only for training data
				train_data = dataset.generate_train_for_one_epoch()
				with PixelBar(f"Epoch {epoch + 1}: ", max=NUM_BATCHES) as bar:
					bar.check_tty = False
					for x_batch, y_batch in train_data:
						global_step+=1
						epoch_step+=1

						#train model runs here
						train_loss_mad, train_loss_mse = self.train_step(x_batch, y_batch)
						total_train_loss_mad += train_loss_mad
						total_train_loss_mse += train_loss_mse

					total_train_loss_mad/=NUM_BATCHES
					total_train_loss_mse/=NUM_BATCHES
					# print("train loss : "+ str(train_loss_mad))
					#Writing tensorboard scalars for train
					with train_summary_writer.as_default():
						tf.summary.scalar('mad_loss', total_train_loss_mad, step=epoch)
						tf.summary.scalar('mse_loss', total_train_loss_mse, step=epoch)
					# print(dataset.val.shape)
					val_data = dataset.generate_val_for_one_epoch()
					val_step = 0
					for x_batch, y_batch in val_data:
						val_step+=1

						#val model runs here
						val_loss_mad, val_loss_mse = self.val_step(x_batch, y_batch)
						total_val_loss_mad += val_loss_mad
						total_val_loss_mse += val_loss_mse

					total_val_loss_mad/=val_step
					total_val_loss_mse/=val_step
					#Writing tensorboad scalars for validation
					with val_summary_writer.as_default():
						tf.summary.scalar('mad_loss', total_val_loss_mad, step=epoch)
						tf.summary.scalar('mse_loss', total_val_loss_mse, step=epoch)


					#bar stuff
					template = "Total training loss: MAD = {:7e} , MSE = {:7e}"
					bar.suffix =  template.format(total_train_loss_mad, total_train_loss_mse)
					bar.next()
					# print(f'\n\nEpoch: {epoch + 1}, Training Loss: {total_train_loss_mad}')
					# print(f'Epoch: {epoch + 1}, Validation Loss: {total_val_loss_mad}\n')

					#Model save condition
					if not self.save_model(self.conf.root, global_step,
						epoch_step, total_val_loss_mad, 0.00001, start_time):
						print(f'Validation loss has not improved from the previous value {Train.VALID_LOSS}')

		except KeyboardInterrupt:
			ckpt_manager.save(int(ckpt.step))
			print("Saved checkpoint for step {}".format(int(ckpt.step)))




if __name__ == '__main__':
	conf = Config('config.yaml')
	dataset = StockDataset(config=conf)
	print("Creating the checkpoint manager")

	model = LSTM_RNN(
		name=conf.name,
		cell_dim=conf.cell_dim,
		layers=conf.layers['count'],
		dropout_rate=conf.layers['dropout_rate'],
		tensorboard=conf.tensorboard,
		lstm_size=conf.lstm['size'],
		device=conf.ops['device'],
		batch_size=conf.ops['batch_size']
		)
	#Creating checkpoint
	checkpoint_dir = os.getcwd() + f"/checkpoints/gap_{conf.data['gap']}/"
	ckpt = tf.train.Checkpoint(step=tf.Variable(0),
		LSTM_RNN=model)
	ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=11)
	#Restoring checkpoint if available
	ckpt.restore(ckpt_manager.latest_checkpoint)
	if ckpt_manager.latest_checkpoint:
		print("Restored from {}".format(ckpt_manager.latest_checkpoint))
	else:
		print("Initializing from scratch.")
	trainer = Train(dataset, model, ckpt, ckpt_manager, 'config.yaml')
