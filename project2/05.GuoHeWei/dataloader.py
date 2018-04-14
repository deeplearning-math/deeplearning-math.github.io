import numpy as np
from config import cfg
from utils import *
import csv
import openpyxl as px
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import random

class Dataloader:
	def __init__(self, hist_size = 0):

		print('==========set up dataloader==========')
		# feature matrix
		self.feature = self.__get_feature(*cfg.dataloader.feature_cell)
		self.preserved_feature = np.copy(self.feature)

		# preprocess the raw data
		self.__preprocessing()

		# train label
		with open(cfg.dataloader.train_label, 'r') as f:
			reader = csv.reader(f)
			self.labels = np.squeeze(np.array(list(reader))[2:, 1:]).astype(np.int)

		# separate data into train, test and validation
		self.num_val = int(cfg.dataloader.num_train_data * cfg.dataloader.num_validation_portion)
		self.num_train = cfg.dataloader.num_train_data - self.num_val
		self.hist_size = hist_size
		self.__separate_data(self.num_train, self.num_val, self.hist_size)
		self.__build_important_sample()

		# dimension of feature vector
		self.feature_dim = self.feature.shape[1]

		self.importance_rate = cfg.dataloader.im_sampling_rate

		print('Train feature shape:', self.train_feature.shape)
		print('Train label shape:', self.train_label.shape)
		print('Validation feature shape:', self.val_feature.shape)
		print('Validation label shape:', self.val_label.shape)
		print('Test feature shape:', self.test_feature.shape)
		print('PCA:', cfg.dataloader.pca.enable_pca)
		print('Remove empty column:', cfg.dataloader.remove_empty_col)
		print('Column-wise normalize:', cfg.dataloader.columnwise_normalize)
		print('categorize column:', cfg.dataloader.categorize_column)
		print('==========done==========')


	def __get_feature(self, up_left, low_right):
		# load excel and store as numpy array
		book = px.load_workbook(cfg.dataloader.feature_raw)
		raw_feature = book.active[up_left: low_right]
		feature_list = []
		for row in raw_feature:
			feature_row = []
			for cell in row:
				if cell.value is None:
					feature_row.append(0)
				else:
					feature_row.append(float(cell.value))

			feature_list.append(feature_row)

		return np.array(feature_list, dtype = np.float32)

	def __separate_data(self, num_train, num_val, hist_size):
		self.train_feature = self.feature[:num_train]
		self.train_label = np.squeeze(self.labels[:num_train])

		self.val_feature = self.feature[num_train:(num_train + num_val)]
		self.val_label = np.squeeze(self.labels[num_train:])

		# set the rest features to be test data
		self.test_feature = self.feature[(num_train + num_val - hist_size + 1):]


	def __build_important_sample(self):
		# get the important (label = 1) training data
		ids = np.where(self.train_label == 1)[0]
		self.important_sample = []
		for i in ids:
			start = max(0, i - self.hist_size)
			self.important_sample.append(self.train_feature[start:i])

		print('----Important Samples----')
		print('data size: ' + str(len(ids)))
		print('hist size: ' + str(self.hist_size))
		print('-------------------------')


	def __preprocessing(self):
		if cfg.dataloader.plot_histogram:
			plot_histogram(self.feature)

		if cfg.dataloader.remove_empty_col:
			self.feature = remove_empty_column(self.feature)

		if cfg.dataloader.columnwise_normalize:
			self.feature = columnwise_normalize(self.feature)

		if cfg.dataloader.pca.enable_pca:
			self.feature = self.__apply_pca(cfg.dataloader.pca.max_to_keep)

		if cfg.dataloader.categorize_column:
			self.feature = colum_catagorize(self.feature)


	def __apply_pca(self, max_to_keep, cal_eg_val = False):
		assert not self.feature is None
		pca = PCA(n_components = max_to_keep)

		# return the histogram of eigenvalue for each feature
		if cal_eg_val:
			centered_matrix = self.feature - self.feature.mean(axis = 1)[:, np.newaxis]
			cov = np.dot(centered_matrix.T, centered_matrix)
			eigvals, _ = np.linalg.eig(cov)
			eigvals = np.sort(eigvals, axis = -1, kind = 'quicksort', order = None)
			# np.histogram(eigvals, bins = eigvals.shape[0])
			# max_margin = 10947137000000
			max_margin = 0
			ind = 0
			for i in range(eigvals.shape[0] - 1):
				if eigvals[i + 1] - eigvals[i] > max_margin:
					max_margin = eigvals[i + 1] - eigvals[i]
					ind = i
			print('min margin ',max_margin, ind)

		feature = pca.fit_transform(self.feature)
		# print(pca.components_.shape)
		return feature

	def next_batch(self, batch_size, important_sample = 0):
		ids_important = []
		if important_sample > 0:
			if random.uniform(0, 1) < self.importance_rate:
				ids_important = list(zip(random.sample(self.important_sample, important_sample), [1 for _ in range(important_sample)]))

		ids = [random.randint(self.hist_size, self.num_train - 1) for _ in range(batch_size)]
		batch = []

		for i in ids:
			start = i - self.hist_size
			batch.append((self.train_feature[start:i], self.train_label[i]))

		batch = ids_important + batch
		random.shuffle(batch)
		
		return batch

if __name__ == '__main__':
	d = Dataloader(hist_size = 5)
	batch = d.next_batch(1, 0)
	# print(batch)
	#print(len(d.feature['C3':'EB985'][0]))

