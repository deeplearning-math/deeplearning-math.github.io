"""Recurrent Models of Visual Attention V. Mnih et al."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf
import numpy as np
from keras.utils.np_utils import to_categorical

from glimpse import GlimpseNet, LocNet
from utils import weight_variable, bias_variable, loglikelihood
from config import Config

from tensorflow.examples.tutorials.mnist import input_data

try:
  xrange
except NameError:
  xrange=range

config = Config()

logging.basicConfig(filename='run-{}.log'.format(config.run_name),level=logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)

rnn_cell = tf.nn.rnn_cell
seq2seq = tf.contrib.legacy_seq2seq


#mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
data = np.load('../data/mnist_digit_sample_8dsistortions9x9.npz')

# the data, shuffled and split between train and test sets

x_train = np.reshape(data['X_train'], (-1, 10000))
y_train = np.reshape(data['y_train'], (-1))
x_va = np.reshape(data['X_valid'], (-1, 10000))
y_va = np.reshape(data['y_valid'], (-1))
x_test = np.reshape(data['X_test'], (-1, 10000))
y_test = np.reshape(data['y_test'], (-1))

#x_train, y_train = mnist.train.next_batch(config.batch_size)

# x_train = x_train.astype('float32')[:32]
# x_va = x_train
# y_va = y_train
# x_test = x_train
# y_test = y_train

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_va.shape[0], 'validation samples')
print(x_test.shape[0], 'test samples')

# config = Config()

input_shape = (config.original_size, config.original_size, 1)

num_epochs = config.step

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

# Monte Carlo sampling, duplicate M times, see Eqn (2)
images_expanded = tf.tile(images_ph, [config.M, 1])
labels_expanded = tf.tile(labels_ph, [config.M])

# Build the aux nets.
with tf.variable_scope('glimpse_net'):
  # gl = GlimpseNet(config, images_ph)
  gl = GlimpseNet(config, images_expanded)
with tf.variable_scope('loc_net'):
  loc_net = LocNet(config)

# number of examples
# N = tf.shape(images_ph)[0]
N = tf.shape(images_expanded)[0]
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
correct_prediction = tf.equal(tf.argmax(softmax,1), labels_expanded)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# average statistics after Monte Carlo sampling "M"
avg_softmax = tf.reshape(softmax, [config.M, -1, config.num_classes])
avg_softmax = tf.reduce_mean(avg_softmax, axis=0) # (B, num_classes)
avg_y_pred = tf.argmax(avg_softmax, axis=1) #(B, )
avg_acc = tf.reduce_mean(tf.cast(tf.equal(avg_y_pred, labels_ph), tf.float32))

# cross-entropy.
xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_expanded)
xent = tf.reduce_mean(xent)

# 0/1 reward.
y_pred = tf.argmax(logits, 1)
reward = tf.cast(tf.equal(y_pred, labels_expanded), tf.float32)
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
training_steps_per_epoch = x_train.shape[0] // config.batch_size
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

# tensorboard logging
tf.summary.scalar("loss", loss)
tf.summary.scalar("reward", reward)
tf.summary.scalar("xent", xent)
tf.summary.scalar("baselines_mse", baselines_mse)
tf.summary.scalar("logllratio", logllratio)
tf.summary.scalar("avg_accuracy", avg_acc)
summary_op = tf.summary.merge_all()

saver = tf.train.Saver()

config_gpu = tf.ConfigProto() 
config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.6 
with tf.Session(config=config_gpu) as sess:
  sess.run(tf.initialize_all_variables())
  writer = tf.summary.FileWriter(logdir="./logs/"+config.run_name, graph=tf.get_default_graph())

  for epoch in xrange(num_epochs):
    num_batches = x_train.shape[0] // config.batch_size
    num_samples = num_batches * config.batch_size
    avg_loss = 0.

    for batch in range(num_batches):
      start = batch * config.batch_size
      end = (batch + 1) * config.batch_size
      images, labels = x_train[start:end], y_train[start:end]

      loc_net.samping = True
      avg_acc_val, softmax_val, adv_val, baselines_val, rewards_val, baselines_mse_val, xent_val, logllratio_val, \
          reward_val, loss_val, lr_val, _, summary_val = sess.run(
              [avg_acc, softmax, advs, baselines, rewards, baselines_mse, xent, logllratio,
               reward, loss, learning_rate, train_op, summary_op],
              feed_dict={
                  images_ph: images,
                  labels_ph: labels
              })
      writer.add_summary(summary_val, epoch * num_batches + batch)

      avg_loss += loss_val / num_batches

      if batch and batch % 100 == 0:
      # if True:
        logging.info('epoch {}: batch: {}/{}'.format(epoch, batch, num_batches - 1))
        logging.info('epoch {}: avg_accuracy: {}'.format(epoch, avg_acc_val))
        logging.info('epoch {}: lr = {:3.6f}'.format(epoch, lr_val))
        logging.info(
            'epoch {}: reward = {:3.4f}\tloss = {:3.4f}\txent = {:3.4f}'.format(
                epoch, reward_val, loss_val, xent_val))
        logging.info('llratio = {:3.4f}\tbaselines_mse = {:3.4f}'.format(
            logllratio_val, baselines_mse_val))
        logging.debug('baselines = {}\trewards = {}'.format(baselines_val, rewards_val))

    # if epoch and epoch % training_steps_per_epoch == 0:
    if True: # print each epoch
      # Evaluation
      for dataset in [(x_va, y_va,'va')]:
        num_batches = dataset[0].shape[0] // config.eval_batch_size
        correct_cnt = 0
        num_samples = num_batches * config.eval_batch_size
        loc_net.sampling = True
        for test_step in xrange(num_batches):
          images, labels = dataset[0][test_step * config.eval_batch_size : (test_step+1) * config.eval_batch_size], dataset[1][test_step * config.eval_batch_size : (test_step+1) * config.eval_batch_size]

          avg_y_pred_val = sess.run(avg_y_pred,
                                 feed_dict={
                                     images_ph: images,
                                     labels_ph: labels
                                 })

          correct_cnt += np.sum(avg_y_pred_val == labels)
        acc = correct_cnt / num_samples

        logging.info('epoch {}: valid accuracy = {}'.format(epoch, acc))

  for dataset in [(x_test, y_test, 'test')]:
    num_batches = dataset[0].shape[0] // config.eval_batch_size
    correct_cnt = 0
    num_samples = num_batches * config.eval_batch_size
    loc_net.sampling = True
    for test_step in xrange(num_batches):
      images, labels = dataset[0][test_step * config.eval_batch_size: (test_step + 1) * config.eval_batch_size], \
                       dataset[1][test_step * config.eval_batch_size: (test_step + 1) * config.eval_batch_size]

      avg_y_pred_val = sess.run(avg_y_pred,
                                feed_dict={
                                  images_ph: images,
                                  labels_ph: labels
                                })

      correct_cnt += np.sum(avg_y_pred_val == labels)
    acc = correct_cnt / num_samples
    logging.info('test accuracy = {}'.format(acc))

  save_path = saver.save(sess, "model-{}.ckpt".format(config.run_name))
  logging.info('Model saved in file: {}'.format(save_path))
