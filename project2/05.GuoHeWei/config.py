from easydict import EasyDict as ed
import os

cfg = ed()
root = os.path.dirname(__file__)

# data path 
cfg.dataloader = ed()
cfg.dataloader.data_path = os.path.join(root, 'data')
cfg.dataloader.feature_raw = os.path.join(cfg.dataloader.data_path, 'feature.xlsx')
cfg.dataloader.feature_cell = ['C3', 'DB985']
cfg.dataloader.feature_saved = None
cfg.dataloader.train_label = os.path.join(cfg.dataloader.data_path, 'train_label.csv')
cfg.dataloader.num_train_data = 683
cfg.dataloader.num_validation_portion = 0.1

# probalility of sampling important data for training
cfg.dataloader.im_sampling_rate = 0.3

# data preprocess config
cfg.dataloader.pca = ed()
cfg.dataloader.pca.enable_pca = False
cfg.dataloader.pca.max_to_keep = 120

# remove empty column
cfg.dataloader.remove_empty_col = True

# normalize each column
cfg.dataloader.columnwise_normalize = False
# cfg.dataloader.apply_columns = [i for i in range()]

# save histogram 
cfg.dataloader.plot_histogram = False

# categorize columns
cfg.dataloader.categorize_column = True

# rnn
cfg.rnn = ed()
cfg.rnn.hidden_size = 512
cfg.rnn.hist_size = 3
cfg.rnn.batch_size = 10
cfg.rnn.train_iter = 100000

cfg.rnn.learning_rate = 0.1
cfg.rnn.decay_step = 5000
cfg.rnn.decay_rate = 0.9
cfg.rnn.clip_norm = 1.25

cfg.rnn.drop_out = False
cfg.rnn.dropout_rate = 0.5

cfg.rnn.batch_normalization = False

cfg.rnn.load_pretrained = True	
cfg.rnn.model_path = os.path.join(root, 'rnn_model')
cfg.rnn.logdir = os.path.join(root, 'rnn_log')


cfg.rnn.validation_freq = 10
cfg.rnn.save_step = 10000
cfg.rnn.num_important_sample = 0

