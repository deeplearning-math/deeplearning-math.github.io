import numpy as np
import tensorflow as tf

import vgg19
import utils

dataset0 = np.load('good0.npy')
dataset0_1 = np.load('bad.npy')
ddd = np.concatenate([dataset0,dataset0_1],axis=0)
dataset_temp = np.concatenate((ddd.reshape(ddd.shape[0],224,224,1),ddd.reshape(ddd.shape[0],224,224,1),ddd.reshape(ddd.shape[0],224,224,1)),axis=3)

dataset1 = np.zeros((dataset0.shape[0],1))
dataset1_2 = np.ones((dataset0_1.shape[0],1))
dataset2_temp = np.concatenate([dataset1,dataset1_2],axis=0)
permute_index = np.random.permutation(dataset_temp.shape[0])

dataset = np.zeros((dataset_temp.shape[0],dataset_temp.shape[1],dataset_temp.shape[2],dataset_temp.shape[3]))
dataset2 = np.zeros((dataset2_temp.shape[0],dataset2_temp.shape[1]))
for i in range(permute_index.shape[0]):
    dataset[i,:,:,:] = dataset_temp[permute_index[i],:,:,:]
    dataset2[i,:] = dataset2_temp[permute_index[i],:]

np.save('./RESULTS/label.npy',dataset2)

# imgcsv = np.loadtxt('rap_sep_imgs.csv',delimiter=',')
# imgcsv2 = imgcsv[:20100]

batch = dataset

f1 = np.zeros((30000,25088))

# with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
with tf.device('/cpu:0'):
    with tf.Session() as sess:
        images = tf.placeholder("float", [100, 224, 224, 3])

        vgg = vgg19.Vgg19()
        with tf.name_scope("content_vgg"):
            vgg.build(images)

        for i in range(300):
            feed_dict = {images: batch[i*100:(i+1)*100]}
            prob = sess.run(vgg.pool5, feed_dict=feed_dict)
            f1[i*100:(i+1)*100] = prob.reshape(100,25088)
            print(i)
        # utils.print_prob(prob[0], './synset.txt')
        # utils.print_prob(prob[1], './synset.txt')

        # prob = sess.run(vgg.pool5, feed_dict=feed_dict)
        #np.save('rap_sep_features.npy',f1)
        np.savetxt('./RESULTS/feature.csv',f1,delimiter=',')



