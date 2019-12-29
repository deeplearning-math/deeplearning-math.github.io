
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

TotTrainData = 7000
TotTestData = 3000
input_dim1 = 224
input_dim2 = 224
output_dim = 1
learn_rate = 5e-4
batch_size = 50
num_batch = int(TotTrainData/batch_size)
num_iters = 10000
keep_prob = 0.9

def defconv2d(x, w1, w2, win, wout, wx, wy, var_scope):
# stride [1, x_movement, y_movement, 1]
# Must have strides[0] = strides[3] = 1
    with tf.variable_scope(var_scope):
        w = tf.Variable(tf.truncated_normal([w1,w2,win,wout], stddev=0.05))
        b = tf.Variable(tf.zeros([wout]))
        # tf.add_to_collection("train_var", w)
        # tf.add_to_collection("train_var", b)
        conv_tensor = (tf.nn.conv2d(x, w, strides=[1,wx,wy,1], padding='SAME'))+b
        batch_normed = tf.contrib.layers.batch_norm(conv_tensor, epsilon=1e-5, is_training=True)#, variables_collections="batch_norm_conv")
        return tf.nn.relu(batch_normed)
    # return conv_tensor

def add_layer(L_Prev, num_nodes_LPrev, num_nodes_LX, activation_LX, var_scope):
    with tf.variable_scope(var_scope):
        Weights_LX = tf.Variable(tf.random_normal([num_nodes_LPrev,num_nodes_LX],stddev=0.5),name = 'w')
        biases_LX = tf.Variable(tf.zeros([1,num_nodes_LX]),name = 'b')
        # tf.add_to_collection("train_var",Weights_LX)
        # tf.add_to_collection("train_var",biases_LX)
        xW_plus_b_LX = tf.matmul(L_Prev,Weights_LX)+biases_LX
        if activation_LX is None:
            LX = xW_plus_b_LX
        else:
            LX = tf.add(xW_plus_b_LX,activation_LX(xW_plus_b_LX))
        return LX

x2 = tf.placeholder(tf.float32, [None, input_dim1, input_dim2, 1])
y2 = tf.placeholder(tf.float32, [None, output_dim])

c1 = defconv2d(x2,3,3,1,16,2,2,'trainv_1') 
c2 = defconv2d(c1,3,3,16,32,2,2,'trainv_2') 
c3 = defconv2d(c2,3,3,32,32,2,2,'trainv_3') 
c4 = defconv2d(c3,3,3,32,64,2,2,'trainv_4')
c5 = defconv2d(c4,3,3,64,128,2,2,'trainv_5')

flat1 = tf.reshape(c5, [-1, 49*128])
L1 = add_layer(flat1,49*128,64,tf.nn.tanh,'trainv_6')
prediction = add_layer(L1,64,output_dim,tf.nn.sigmoid,'trainv_7')

# the error between prediction and real data
loss = tf.reduce_mean(tf.square(y2-prediction))
optimizer = tf.train.AdamOptimizer(learn_rate).minimize(loss)

saver = tf.train.Saver()

dataset0 = np.load('good0.npy')
dataset0_1 = np.load('bad.npy')
dataset_temp = np.concatenate([dataset0,dataset0_1],axis=0)

dataset1 = np.zeros((dataset0.shape[0],1))
dataset1_2 = np.ones((dataset0_1.shape[0],1))
dataset2_temp = np.concatenate([dataset1,dataset1_2],axis=0)
permute_index = np.random.permutation(dataset_temp.shape[0])

dataset = np.zeros((dataset_temp.shape[0],dataset_temp.shape[1],dataset_temp.shape[2]))
dataset2 = np.zeros((dataset2_temp.shape[0],dataset2_temp.shape[1]))
for i in range(permute_index.shape[0]):
    dataset[i,:,:] = dataset_temp[permute_index[i],:,:]
    dataset2[i,:] = dataset2_temp[permute_index[i],:]

dataset_reshaped = dataset.reshape((dataset.shape[0],input_dim1,input_dim2,1))
x_data = dataset_reshaped[0:TotTrainData,:,:,:]
y_data = dataset2[0:TotTrainData,:].reshape(TotTrainData,output_dim)
x_data_pred = dataset_reshaped[0:1000,:,:,:]
y_data_pred = dataset2[0:1000,:].reshape(1000,output_dim)
x_data_test = dataset_reshaped[TotTrainData:TotTrainData+TotTestData,:,:,:]
y_data_test = dataset2[TotTrainData:TotTrainData+TotTestData,:].reshape(TotTestData,output_dim)

iii = 0
losses = []

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print("Training started")
    if False:
        saver.restore(sess,'./save/model')

    for step in range(num_iters):
        for batch_i in range(num_batch):
            x_batch = x_data[batch_i*batch_size:(batch_i+1)*batch_size,:,:,:]
            y_batch = y_data[batch_i*batch_size:(batch_i+1)*batch_size,:]
            sess.run(optimizer,feed_dict={x2:x_batch,y2:y_batch})
            losses.append(np.mean(sess.run(loss,feed_dict={x2:x_batch,y2:y_batch})))

        curr_loss = sess.run(loss,feed_dict={x2:x_data_test,y2:y_data_test})
        predval = sess.run(prediction,feed_dict={x2:x_data_test})
        predval = predval>0.5
        acc = np.mean(np.abs(predval-y_data_test))
        print('iter:',step,' loss:',curr_loss, ' acc:',acc)
        if (step+1)%5 == 0:
            saver.save(sess,'save/model')

