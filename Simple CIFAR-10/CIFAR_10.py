from keras.datasets import cifar10
import keras
from keras import backend as K
import numpy as np
from keras.utils import np_utils
from Freeze_Graph_Script import freeze_graph
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.constraints import maxnorm

# from matplotlib import pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# plt.imshow(x_test[0])
# plt.show()
# print(y_test[0])


# Function to reshape each image (32 x 32 x 3) into 1 long array (1 x 3072)
def reshape_image(input_array):
    output_array = []
    for image in input_array:
        output_array.append(image.reshape(-1))
    return np.asarray(output_array)


# Convert the images into floats and normalize them so each value is >= 0 and <= 1
train_images = x_train.astype('float32') / 255.0
test_images = x_test.astype('float32') / 255.0
# Flatten each of the image arrays (easier to feed into android this way)
train_images = reshape_image(train_images)
test_images = reshape_image(test_images)
# Change each of the labels into a categorical format for easy feed into our model
train_labels = np_utils.to_categorical(y_train)
test_labels = np_utils.to_categorical(y_test)

# Sets the learning phase so that we don't have to enter a value for dropout in android
K.set_learning_phase(1)

# Set up the sequential type model (just add layers in the order you want them to be executed
model = Sequential()
# Reshape the input into a 32 x 32 matrix with 3 color channels for more accurate analysis
model.add(Reshape((32, 32, 3), input_shape=(3072,)))
# Associate 3 x 3 patches of each image with 32 features (because we are using float32) with relu activation function
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
# Get the max value of each 2 x 2 block and discard the rest, thus changing image size to 16 x 16
model.add(MaxPooling2D(pool_size=(2, 2)))
# Flatten the matrix into an array
model.add(Flatten())
# Dense layer creates 512 connected neurons (32 x 16) with relu activation function
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
# Dropout layer prevents overfitting by ignoring outputs from some neurons
model.add(Dropout(0.5))
# Last dense layer sorts the features into 10 possible outputs with softmax activation
model.add(Dense(10, activation='softmax'))

# Specify loss function, optimizer, and which metrics we want to keep track of
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=30, batch_size=32)

frozen_graph = freeze_graph(K.get_session(), output_names=[model.output.op.name])
tf.train.write_graph(frozen_graph, '.', 'cifar10.pb', as_text=False)
