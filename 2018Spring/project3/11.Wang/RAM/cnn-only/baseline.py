from __future__ import print_function
import keras
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.utils.np_utils import to_categorical
import numpy as np

sess = tf.Session()
K.set_session(sess)

batch_size = 128
num_classes = 10
epochs = 100
run_name = '{}epochs'.format(epochs)

# input image dimensions
img_rows, img_cols = 100, 100

data = np.load('../data/mnist_digit_sample_8dsistortions9x9.npz')

# the data, shuffled and split between train and test sets
x_train = np.expand_dims(data['X_train'], axis=-1)
y_train = to_categorical(np.reshape(data['y_train'], (-1)))
x_va = np.expand_dims(data['X_valid'], axis=-1)
y_va = to_categorical(np.reshape(data['y_valid'], (-1)))
x_test = np.expand_dims(data['X_test'], axis=-1)
y_test = to_categorical(np.reshape(data['y_test'], (-1)))

input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_va = x_va.astype('float32')
x_test = x_test.astype('float32')

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_va.shape[0], 'validation samples')
print(x_test.shape[0], 'test samples')

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs/'+run_name, histogram_freq=0, write_graph=True, write_images=True)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_va, y_va),
          callbacks=[tbCallBack]
         )
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# serialize model to JSON
model_json = model.to_json()
with open("model-{}.json".format(run_name), "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model-{}.h5".format(run_name))
print("Saved model to disk")
