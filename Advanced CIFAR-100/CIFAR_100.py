from keras.datasets import cifar100
import tensorflow as tf
from keras import backend as K
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Reshape, Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.constraints import maxnorm
from Freeze_Graph import freeze_session
K.set_learning_phase(1)


def reshape_inputs(input_array):
    output_array = []
    for i in input_array:
        output_array.append(i.reshape(-1))
    return np.asarray(output_array)


(x_train, y_train), (x_test, y_test) = cifar100.load_data()

new_x_train = x_train.astype('float32') / 255.0
new_x_test = x_test.astype('float32') / 255.0
new_x_train = reshape_inputs(new_x_train)
new_x_test = reshape_inputs(new_x_test)
new_y_train = np_utils.to_categorical(y_train)
new_y_test = np_utils.to_categorical(y_test)

model = Sequential()
model.add(Reshape((32, 32, 3), input_shape=(3072,)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(100, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])

saver = tf.train.Saver()

model.fit(new_x_train, new_y_train, epochs=50, batch_size=32)

frozen_graph = freeze_session(K.get_session(), output_names=[model.output.op.name])
tf.train.write_graph(frozen_graph, '.', 'cifar100.pb', as_text=False)
