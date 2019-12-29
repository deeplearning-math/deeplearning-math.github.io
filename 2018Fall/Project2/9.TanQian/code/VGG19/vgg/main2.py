import numpy as np
import tensorflow as tf

import vgg19
import utils

ddd = np.load('test.npy')
dataset_temp = np.concatenate((ddd.reshape(ddd.shape[0],224,224,1),ddd.reshape(ddd.shape[0],224,224,1),ddd.reshape(ddd.shape[0],224,224,1)),axis=3)

dataset = dataset_temp

# imgcsv = np.loadtxt('rap_sep_imgs.csv',delimiter=',')
# imgcsv2 = imgcsv[:20100]

batch = dataset

f1 = np.zeros((3000,25088))

# with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
with tf.device('/cpu:0'):
    with tf.Session() as sess:
        images = tf.placeholder("float", [100, 224, 224, 3])

        vgg = vgg19.Vgg19()
        with tf.name_scope("content_vgg"):
            vgg.build(images)

        for i in range(30):
            feed_dict = {images: batch[i*100:(i+1)*100]}
            prob = sess.run(vgg.pool5, feed_dict=feed_dict)
            f1[i*100:(i+1)*100] = prob.reshape(100,25088)
            print(i)
        # utils.print_prob(prob[0], './synset.txt')
        # utils.print_prob(prob[1], './synset.txt')

        # prob = sess.run(vgg.pool5, feed_dict=feed_dict)
        #np.save('rap_sep_features.npy',f1)
        np.savetxt('./RESULTS2/feature.csv',f1,delimiter=',')



