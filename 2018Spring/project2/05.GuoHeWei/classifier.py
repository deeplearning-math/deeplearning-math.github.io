import numpy as np
import os
from sklearn import svm
from dataloader import Dataloader
import tensorflow as tf
from utils import generate_submission
from sklearn.metrics import roc_auc_score
from config import cfg

class SVM:
	def __init__(self, dataloader):
		self.train_feature_vec = dataloader.train_feature
		self.train_label = dataloader.train_label
		self.val_feature_vec = dataloader.val_feature
		self.val_label = dataloader.val_label
		self.test_feature = dataloader.test_feature

		self.setup_predictor()

	def setup_predictor(self):
		self.predictor = svm.SVC(probability = True)
		self.predictor.fit(self.train_feature_vec, self.train_label)  

	def validation(self):
		return self.predictor.predict(self.val_feature_vec)

	def test(self):
		result = self.predictor.predict_proba(self.test_feature)
		generate_submission(result, '.', '.')

class RNN:
	def __init__(self, dataloader,modeltype='gru'):

		print('=====Initializing rnn=====')
		# training config
		self.batch_size = cfg.rnn.batch_size
		self.hist_size = cfg.rnn.hist_size
		self.train_iter = cfg.rnn.train_iter
		self.learning_rate = cfg.rnn.learning_rate
		self.decay_step = cfg.rnn.decay_step 
		self.decay_rate = cfg.rnn.decay_rate
		self.clip_norm = cfg.rnn.clip_norm


		# network input
		self.input = tf.placeholder(tf.float32, shape = [None, self.hist_size, dataloader.feature_dim])
		self.labels = tf.placeholder(tf.float32, shape = [None])



		if modeltype== 'gru':
			# network body
			self.gru_cell = tf.contrib.rnn.GRUCell(cfg.rnn.hidden_size)
			rnn_out, self.last_state = tf.nn.dynamic_rnn(self.gru_cell, self.input, dtype=tf.float32)
		elif modeltype=='lstm':
			# network body
			self.lstm_cell = tf.contrib.rnn.LSTMCell(cfg.rnn.hidden_size)
			lstm_out, tup = tf.nn.dynamic_rnn(self.lstm_cell, self.input, dtype=tf.float32)
			_,self.last_state = tup
		elif modeltype=='vanilla':
			self.vanillarnn_cell = tf.contrib.rnn.BasicRNNCell(cfg.rnn.hidden_size)
			rnn_out, self.last_state = tf.nn.dynamic_rnn(self.vanillarnn_cell,self.input,dtype=tf.float32)



		self.enable_dropout = cfg.rnn.drop_out
		if self.enable_dropout:
			self.keep_prob = tf.placeholder(tf.float32)
			self.last_state = tf.nn.dropout(self.last_state, self.keep_prob)
			self.dropout_rate = cfg.rnn.dropout_rate

		# network output
		self.logit = tf.squeeze(tf.layers.dense(self.last_state, 1))
		self.pred = tf.nn.sigmoid(self.logit)

		# loss function
		self.loss_func = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.logit, labels = self.labels))
		self.opt = self.__get_optimizer()

		# other config
		self.dataloader = dataloader
		self.num_important_sample = cfg.rnn.num_important_sample

		self.sess = tf.InteractiveSession()
		self.sess.run(tf.global_variables_initializer())
		self.validation_freq = cfg.rnn.validation_freq
		self.save_step = cfg.rnn.save_step
		self.model_path = cfg.rnn.model_path + '/' + modeltype

		if not os.path.exists(self.model_path):
			os.makedirs(self.model_path)
			
		# tensorboard
		tf.summary.scalar('rnn_loss', self.loss_func)
		self.sum_writer = tf.summary.FileWriter(cfg.rnn.logdir + modeltype, self.sess.graph)
		self.summary_merge = tf.summary.merge_all()

		# load pretrained model
		self.saver = tf.train.Saver()
		if cfg.rnn.load_pretrained:
			checkpoint = tf.train.get_checkpoint_state(self.model_path)
			if checkpoint and checkpoint.model_checkpoint_path:
				self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
				print('Successfully loaded:', checkpoint.model_checkpoint_path)
			else:
				print('Could not find old network weights')
		print('=====Done=====')


	def __get_optimizer(self):
		assert not self.loss_func is None

		global_step = tf.Variable(0, trainable = False)
		learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, 
	    										   self.decay_step, self.decay_rate, staircase = True)
		optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		gradients, v = zip(*optimizer.compute_gradients(self.loss_func))
		gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)
		optimizer = optimizer.apply_gradients(zip(gradients, v), global_step = global_step)

		return optimizer

	def train(self):
		print('=====Training Start=====')
		print('Initial learning rate:', self.learning_rate)
		print('History size:', self.hist_size)
		print('Number of iteration:', self.train_iter)
		if self.enable_dropout:
			print('Dropout rate: ', self.dropout_rate)

		for i in range(1, self.train_iter):
			batch = self.dataloader.next_batch(self.batch_size, self.num_important_sample)
			inputs, labels = zip(*batch)
			feed_dict = {self.input: list(inputs), self.labels: list(labels)}
			if self.enable_dropout:
				feed_dict[self.keep_prob] = self.dropout_rate

			# train
			_, log, loss = self.sess.run([self.opt, self.summary_merge, self.loss_func], feed_dict = feed_dict)
			if i % 100 == 0:
				print('Iteration #%i, loss:%s' %(i, loss))

			# write to tensorboard
			self.sum_writer.add_summary(log, i)
			# save model
			if i % self.save_step == 0:
				self.saver.save(self.sess, self.model_path, global_step = i)
				print('Model saved in %ith iteration' %i)


			# check over fit
			if i % 5000 == 0:
				batch = []
				for i in range(self.dataloader.train_feature.shape[0] - self.hist_size + 1):
					batch.append(self.dataloader.train_feature[i:(i + self.hist_size)])

				ground_truth = self.dataloader.train_label[(self.hist_size - 1):]
				test_feature = np.array(batch)

				feed_dict = {self.input:test_feature}
				if self.enable_dropout:
					feed_dict[self.keep_prob] = 1

				result = self.sess.run(self.pred, feed_dict = feed_dict)
				auc = roc_auc_score(ground_truth, result)

				print('The overfitted accuracy is:', auc)


		print('=====Training Done!=====')

	def __make_bound(self, data):
		assert len(data.shape) == 1
		for i in range(data.shape[0]):
			if data[i] < 0.005:
				data[i] = 0.005
			elif data[i] > 0.995:
				data[i] = 0.995

	def test(self):
		print('=====Generating test result=====')
		# test_feature = np.expand_dims(self.dataloader.test_feature, axis = 0)
		batch = []
		for i in range(self.dataloader.test_feature.shape[0] - self.hist_size + 1):
			batch.append(self.dataloader.test_feature[i:(i + self.hist_size)])

		test_feature = np.array(batch)

		feed_dict = {self.input:test_feature}
		if self.enable_dropout:
			feed_dict[self.keep_prob] = 1

		result = self.sess.run(self.pred, feed_dict = feed_dict)
		self.__make_bound(result)

		generate_submission(result, '.', '.')
		print('=====Done=====')

	def validation(self):
		batch = []
		for i in range(self.dataloader.val_feature.shape[0] - self.hist_size + 1):
			batch.append(self.dataloader.test_feature[i:(i + self.hist_size)])

		ground_truth = self.dataloader.val_label[(self.hist_size - 1):]
		test_feature = np.array(batch)

		feed_dict = {self.input:test_feature}
		if self.enable_dropout:
			feed_dict[self.keep_prob] = 1

		result = self.sess.run(self.pred, feed_dict = feed_dict)

		# bound the prediction value
		self.__make_bound(result)

		auc = roc_auc_score(ground_truth, result)

		print('The validation accuracy is:', auc)

if __name__ == '__main__':
	dataloader = Dataloader()
	classifier = RNN(dataloader,modeltype='gru')
	print('ok')
