import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_submission(result, path, save_path):


	assert os.path.isdir(path)
	assert os.path.isdir(save_path)

	output = pd.read_csv(os.path.join(path, 'test_label_sample.csv'))

	with open (os.path.join(save_path, 'submission.csv'),'w') as file:
		file.write('date,label\n')
		for i in range(300):
			file.write(str(output['date'][i]) + ',' + str(result[i]) + '\n')

def remove_empty_column(data):
	assert len(data.shape) == 2
	return data[:, ~np.all(data == 0, axis = 0)]

def columnwise_normalize(data, apply_columns = None):
	assert len(data.shape) == 2

	if not apply_columns is None:
		assert len(apply_columns) > 0
		apply_columns = sorted(apply_columns)
		assert data.shape[1] > apply_columns[-1]

		base = np.ones(data.shape[1])
		base[apply_columns] = data.max(axis = 0)[apply_columns]
		return data / base
	else:
		return data / data.max(axis = 0)
	
def colum_catagorize(data, lower_bd = 0.1, upper_bd = 0.9, size_bd = 20):
	new_data = np.zeros_like(data, dtype = np.float)

	# loop columns
	for i in range(data.shape[1]):
		col = data[:, i]
		# remove zero elements
		reduced_col = col[col > 0]	 
		size = reduced_col.shape[0]

		if size <= size_bd:
			# low occurance, use 0 or 1
			for idx in range(col.shape[0]):
				if col[idx] > 0:
					new_data[idx, i] = 1
		else:
			# high occurance, use -1 to denote lower 10 percentile
			# use 1 to denote upper 10 precentile, 0 for other cases
			reduced_col = np.sort(reduced_col)
			lower = reduced_col[int(size * lower_bd)]
			upper = reduced_col[int(size * upper_bd)]
			for idx in range(col.shape[0]):
				if col[idx] <= lower and col[idx] > 0:
					new_data[idx, i] = -1
				elif col[idx] >= upper:
					new_data[idx, i] = 2
				elif col[idx] < upper and col[idx] > lower:
					new_data[idx, i] = 1
	return new_data

def plot_histogram(data, num_bins = 30, remove_zero = True, plot = True):
	print('----ploting histogram----')

	if remove_zero:
		folder = 'histogram_non_zero'
	else:
		folder = 'histogram'

	for i in range(data.shape[1]):
		print('ploting column %i' %(i + 1))
		path = os.path.join(folder, 'column-' + str(i + 1))
		cur_col = data[:, i]
		if remove_zero:
			cur_col = cur_col[cur_col > 0]
		plt.hist(cur_col, bins = num_bins)
		plt.ylabel('#data');
		if plot:
			plt.show()
		else:
			plt.savefig(path)
		plt.clf()
	print('----ploting done----')

if __name__ == '__main__':
	# data = np.array([[1, 10], [2, 9], [3, 8], [4, 7], [5, 6], [6, 5], [7, 4], [8, 3], [9, 2], [10, 1], [0, 2], [4, 0]])
	data = np.array([[1, 10], [0, 0], [3, 8], [4, 7], [5, 6], [6, 5], [0, 0], [8, 3], [0, 0], [10, 1], [0, 2], [4, 0]])
	print(data)
	print(colum_catagorize(data))


