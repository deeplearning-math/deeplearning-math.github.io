'''
the code is modified from 
https://github.com/johnsmithm/multi-heads-attention-image-classification/blob/master/multi-heads-attention-mnist.py 
'''

from keras.layers import Dense, Dropout,  Conv2D, Input, Lambda, Flatten, TimeDistributed
from keras.layers import Add, Reshape, MaxPooling2D, Concatenate, Embedding, RepeatVector
from keras.models import Model
from keras import backend as K

import numpy as np
from keras.datasets import mnist, fashion_mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.engine.topology import Layer
import tensorflow as tf
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

def MultiHeadsAttModel(l=8*8, d=512, dv=64, dout=512, nv = 8 ):

    v1 = Input(shape = (l, d))
    q1 = Input(shape = (l, d))
    k1 = Input(shape = (l, d))

    v2 = Dense(dv*nv, activation = "relu")(v1)
    q2 = Dense(dv*nv, activation = "relu")(q1)
    k2 = Dense(dv*nv, activation = "relu")(k1)

    v = Reshape([l, nv, dv])(v2)
    q = Reshape([l, nv, dv])(q2)
    k = Reshape([l, nv, dv])(k2)
        
    att = Lambda(lambda x: K.batch_dot(x[0],x[1] ,axes=[-1,-1]) / np.sqrt(dv),
                 output_shape=(l, nv, nv))([q,k])# l, nv, nv
    att = Lambda(lambda x:  K.softmax(x) , output_shape=(l, nv, nv))(att)

    out = Lambda(lambda x: K.batch_dot(x[0], x[1],axes=[4,3]),  output_shape=(l, nv, dv))([att, v])
    out = Reshape([l, d])(out)
    
    out = Add()([out, q1])

    out = Dense(dout, activation = "relu")(out)

    return  Model(inputs=[q1,k1,v1], outputs=out)

class NormL(Layer):

    def __init__(self, **kwargs):
        super(NormL, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.a = self.add_weight(name='kernel', 
                                      shape=(1,input_shape[-1]),
                                      initializer='ones',
                                      trainable=True)
        self.b = self.add_weight(name='kernel', 
                                      shape=(1,input_shape[-1]),
                                      initializer='zeros',
                                      trainable=True)
        super(NormL, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        eps = 0.000001
        mu = K.mean(x, keepdims=True, axis=-1)
        sigma = K.std(x, keepdims=True, axis=-1)
        ln_out = (x - mu) / (sigma + eps)
        return ln_out*self.a + self.b

    def compute_output_shape(self, input_shape):
        return input_shape
    
if __name__ == '__main__':   
    
    import argparse

    parser = argparse.ArgumentParser(description='Train CNN with Multi-Head')
    parser.add_argument('-ds','--dataset', type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('-mh','--multihead', action='store_true')

    args = vars(parser.parse_args())
    print(args)
    
    fname = args['dataset']
    fname += "-multiHeadModel_weights.h5" if args['multihead'] else "-Model_weights.h5"
    
    nb_classes = 10

    # the data, shuffled and split between tran and test sets
    if args['dataset']=='mnist':   
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(60000, 28,28,1)
        X_test = X_test.reshape(10000, 28,28,1)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255
        X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    elif args['dataset']=='fashion-mnist': 
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        X_train = X_train.reshape(60000, 28,28,1)
        X_test = X_test.reshape(10000, 28,28,1)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255
        X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    elif args['dataset']=='cluttered-mnist': 
        mnist_cluttered = np.load('./mnist_sequence1_sample_5distortions5x5.npz')
        X_train = mnist_cluttered['X_train']
        X_train = X_train.reshape(-1, 40,40,1)
        y_train = mnist_cluttered['y_train']
        X_dev = mnist_cluttered['X_valid']
        X_dev = X_dev.reshape(-1, 40,40,1)
        y_dev = mnist_cluttered['y_valid']
        X_test = mnist_cluttered['X_test']
        X_test = X_test.reshape(-1, 40,40,1)
        y_test = mnist_cluttered['y_test']

    print("Training matrix shape", X_train.shape)
    print("Development matrix shape", X_dev.shape)
    print("Testing matrix shape", X_test.shape)

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_dev = np_utils.to_categorical(y_dev, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    
    inp = Input(shape = X_train[0].shape)
    x = Conv2D(32,(2,2),activation='relu', padding='same')(inp)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64,(2,2),activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Conv2D(64*3,(2,2),activation='relu')(x)
    
    if args['multihead']:
        temp = int(x.shape[1])
        x = Reshape([temp*temp,64*3])(x)    
        att = MultiHeadsAttModel(l=temp*temp, d=64*3 , dv=8*3, dout=32, nv = 8 )
        x = att([x,x,x])
        x = Reshape([temp,temp,32])(x)   
        x = NormL()(x)
    
    x = Flatten()(x) 
    x = Dense(256, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    model = Model(inputs=inp, outputs=x)
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    
    if args['test']:
        model.load_weights('models/'+fname)
        results = model.evaluate(X_test, Y_test)
        print('\n results',results)
    else:
        tbCallBack = TensorBoard(log_dir='./Graph/mhatt1', histogram_freq=0, write_graph=True, write_images=True)
        esCallBack = EarlyStopping(monitor='val_acc', min_delta=0, patience=2, verbose=0, mode='max')
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=0, min_lr=0.0001)

        model.fit(X_train, Y_train,
                  batch_size=32, 
                  epochs=100,
                  verbose=1,          
                  validation_data=(X_dev, Y_dev),
                  callbacks=[tbCallBack, esCallBack, reduce_lr])
        
        model.save_weights('models/'+fname)
    
    
    
    
    
    