"""Recurrent Models of Visual Attention V. Mnih et al."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf
import numpy as np

from glimpse import GlimpseNet, LocNet
from utils import weight_variable, bias_variable, loglikelihood
from config import Config

from tensorflow.examples.tutorials.mnist import input_data

try:
  xrange
except NameError:
  xrange=range

logging.getLogger().setLevel(logging.INFO)

rnn_cell = tf.nn.rnn_cell
seq2seq = tf.contrib.seq2seq

import pickle

def truncate_labels(labels):
    """
    (1) replacing row[0] by 10, and move it to the last of row
    (2) replace the second 10 by -1 row wise
    """
    def do_one_row(row):
        erase = False
        for i, _ in enumerate(row):
            if erase:
                row[i] = -1
            else:
                if row[i] == 10:
                    erase = True
        return row

    ret = np.copy(labels)
    ret = repair_labels(ret)
    return np.apply_along_axis(do_one_row, axis=1, arr=ret)

def repair_labels(labels):
    """
    replacing row[0] by 10, and move it to the last of row
    :param labels:
    :return:
    """
    ret = np.copy(labels)
    ret[:, 0] = 10  # overwrite length to be stop seq
    ret = np.roll(ret, -1, axis=1)  # move first to last
    return ret

def mask_labels(labels):
    """
    (1) replacing row[0] by 10, and move it to the last of row
    (2) replace the second 10 by -1 row wise
    """
    def do_one_row(row):
        erase = False
        for i, _ in enumerate(row):
            if erase:
                row[i] = 0
            else:
                if row[i] == 10:
                    erase = True
                row[i] = 1
        return row

    ret = np.copy(labels)
    return np.apply_along_axis(do_one_row, axis=1, arr=ret)

print('Loading pickled data...')

pickle_file = 'SVHN.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    X_train = save['train_dataset']
    Y_train = save['train_labels']
    Y_train = repair_labels(Y_train)
    Y_train_mask = mask_labels(Y_train)
    # X_val = save['valid_dataset']
    # Y_val = save['valid_labels']
    # Y_val = repair_labels(Y_val)
    # Y_val_mask = mask_labels(Y_val)
    # X_test = save['test_dataset']
    # Y_test = save['test_labels']
    # Y_test = repair_labels(Y_test)
    # Y_test_mask = mask_labels(Y_test)
    del save
    print('Training data shape:', X_train.shape)
    print('Training label shape:',Y_train.shape)
    # print('Validation data shape:', X_val.shape)
    # print('Validation label shape:', Y_val.shape)
    # print('Test data shape:', X_test.shape)
    # print('Test label shape:', Y_test.shape)

print('Data successfully loaded!')

# TO BE REMOVED

X_train = X_train[:128]
Y_train = Y_train[:128]

# mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

config = Config()
n_steps = config.step

loc_mean_arr = []
sampled_loc_arr = []


def get_next_input(output, i):
  loc, loc_mean = loc_net(output)
  gl_next = gl(loc)
  loc_mean_arr.append(loc_mean)
  sampled_loc_arr.append(loc)
  return gl_next

# placeholders
images_ph = tf.placeholder(tf.float32,
                           [None, config.original_size * config.original_size *
                            config.num_channels])
labels_ph = tf.placeholder(tf.int64, [None])

# Build the aux nets.
with tf.variable_scope('glimpse_net'):
  gl = GlimpseNet(config, images_ph)
with tf.variable_scope('loc_net'):
  loc_net = LocNet(config)

# number of examples
N = tf.shape(images_ph)[0]
init_loc = tf.random_uniform((N, 2), minval=-1, maxval=1)
init_glimpse = gl(init_loc)
# Core network.
lstm_cell = rnn_cell.LSTMCell(config.cell_size, state_is_tuple=True)
init_state = lstm_cell.zero_state(N, tf.float32)
inputs = [init_glimpse]
inputs.extend([0] * (config.num_glimpses))
outputs, _ = seq2seq.rnn_decoder(
    inputs, init_state, lstm_cell, loop_function=get_next_input)

# Time independent baselines
with tf.variable_scope('baseline'):
  w_baseline = weight_variable((config.cell_output_size, 1))
  b_baseline = bias_variable((1,))
baselines = []
for t, output in enumerate(outputs[1:]):
  baseline_t = tf.nn.xw_plus_b(output, w_baseline, b_baseline)
  baseline_t = tf.squeeze(baseline_t)
  baselines.append(baseline_t)
baselines = tf.stack(baselines)  # [timesteps, batch_sz]
baselines = tf.transpose(baselines)  # [batch_sz, timesteps]

# Take the last step only.
output = outputs[-1]
# Build classification network.
with tf.variable_scope('cls'):
  w_logit = weight_variable((config.cell_output_size, config.num_classes))
  b_logit = bias_variable((config.num_classes,))
logits = tf.nn.xw_plus_b(output, w_logit, b_logit)
softmax = tf.nn.softmax(logits)

# cross-entropy.
xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph)
xent = tf.reduce_mean(xent)
pred_labels = tf.argmax(logits, 1)
# 0/1 reward.
reward = tf.cast(tf.equal(pred_labels, labels_ph), tf.float32)
rewards = tf.expand_dims(reward, 1)  # [batch_sz, 1]
rewards = tf.tile(rewards, (1, config.num_glimpses))  # [batch_sz, timesteps]
logll = loglikelihood(loc_mean_arr, sampled_loc_arr, config.loc_std)
advs = rewards - tf.stop_gradient(baselines)
logllratio = tf.reduce_mean(logll * advs)
reward = tf.reduce_mean(reward)

baselines_mse = tf.reduce_mean(tf.square((rewards - baselines)))
var_list = tf.trainable_variables()
# hybrid loss
loss = -logllratio + xent + baselines_mse  # `-` for minimize
grads = tf.gradients(loss, var_list)
grads, _ = tf.clip_by_global_norm(grads, config.max_grad_norm)

# learning rate
global_step = tf.get_variable(
    'global_step', [], initializer=tf.constant_initializer(0), trainable=False)
training_steps_per_epoch = X_train.shape[0] // config.batch_size
starter_learning_rate = config.lr_start
# decay per training epoch
learning_rate = tf.train.exponential_decay(
    starter_learning_rate,
    global_step,
    training_steps_per_epoch,
    0.97,
    staircase=True)
learning_rate = tf.maximum(learning_rate, config.lr_min)
opt = tf.train.AdamOptimizer(learning_rate)
train_op = opt.apply_gradients(zip(grads, var_list), global_step=global_step)

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  for i in xrange(n_steps):
    # images, labels = mnist.train.next_batch(config.batch_size)
    images, labels = X_train, Y_train
    # duplicate M times, see Eqn (2)
    images = np.tile(images, [config.M, 1, 1])
    labels = np.tile(labels, [config.M, 1])
    loc_net.samping = True
    loc_w_val, adv_val, baselines_val, rewards_val, baselines_mse_val, xent_val, logllratio_val, \
        reward_val, loss_val, lr_val, _ = sess.run(
            [loc_net.w, advs, baselines, rewards, baselines_mse, xent, logllratio,
             reward, loss, learning_rate, train_op],
            feed_dict={
                images_ph: images,
                labels_ph: labels
            })
    # if i and i % 100 == 0:
    if True:
      logging.info('step {}: lr = {:3.6f}'.format(i, lr_val))
      logging.info(
          'step {}: reward = {:3.4f}\tloss = {:3.4f}\txent = {:3.4f}'.format(
              i, reward_val, loss_val, xent_val))
      logging.info('advs = {}\tllratio = {:3.4f}\tbaselines = {}\trewards = {}\tbaselines_mse = {:3.4f}'.format(
          adv_val,logllratio_val, baselines_val, rewards_val, baselines_mse_val))
      logging.info('log_w = {}'.format(loc_w_val))

    # if i and i % training_steps_per_epoch == 0:
    if True:
      # Evaluation
      for _ in range(1):
        steps_per_epoch = X_train.shape[0] // config.eval_batch_size
        correct_cnt = 0
        num_samples = steps_per_epoch * config.batch_size
        loc_net.sampling = True
        for test_step in xrange(steps_per_epoch):
          # images, labels = dataset.next_batch(config.batch_size)
          images, labels = X_train, Y_train
          labels_bak = labels
          # Duplicate M times
          images = np.tile(images, [config.M, 1, 1])
          labels = np.tile(labels, [config.M, 1])
          softmax_val = sess.run(softmax,
                                 feed_dict={
                                     images_ph: images,
                                     labels_ph: labels
                                 })
          softmax_val = np.reshape(softmax_val,
                                   [config.M, -1, config.num_classes])
          softmax_val = np.mean(softmax_val, 0)
          pred_labels_val = np.argmax(softmax_val, 1)
          pred_labels_val = pred_labels_val.flatten()
          correct_cnt += np.sum(pred_labels_val == labels_bak)
        acc = correct_cnt / num_samples
        if dataset == mnist.validation:
          logging.info('valid accuracy = {}'.format(acc))
        else:
          logging.info('test accuracy = {}'.format(acc))
