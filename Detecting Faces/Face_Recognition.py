from keras.datasets import cifar10
import glob
from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.layers import Reshape, Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras import backend as K
K.set_image_dim_ordering('th')
K.set_learning_phase(1)
from Freeze_Graph_Script import freeze_session
import tensorflow as tf


# Function to load the first 60 images from our cifar 10 data set and use these as the non face images
def load_non_face_images():
    (_, _), (cifar_10_images, _) = cifar10.load_data()
    cifar_10_images = cifar_10_images[:60]
    cifar_10_images = cifar_10_images.astype('float32') / 255.0
    return cifar_10_images


# Function to load about 60 face images from our Face_Images dir and format them
def load_face_images(dir_name):
    output_images = []
    for image_name in glob.glob(dir_name + '/*'):
        image = Image.open(image_name).resize((32, 32))
        image_data = np.asarray(image, dtype='float32') / 255.0
        output_image_data = np.transpose(image_data, (2, 0, 1))
        output_images.append(output_image_data)
    return np.asarray(output_images)


# Function to divide data into training and testing data and also assign labels
def separate_data(face_images, non_face_images):
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    for i in range(10):
        test_images.append(face_images[i])
        test_labels.append([1, 0])
        test_images.append(non_face_images[i])
        test_labels.append([0, 1])
    for i in range(len(face_images) - 10):
        train_images.append(face_images[i + 10])
        train_labels.append([1, 0])
        train_images.append(non_face_images[i + 10])
        train_labels.append([0, 1])
    return np.asarray(train_images), np.asarray(train_labels), np.asarray(test_images), np.asarray(test_labels)


# Function to flatten image data arrays for easy input into model
def reshape_image(input_array):
    output_array = []
    for image in input_array:
        output_array.append(image.reshape(-1))
    return np.asarray(output_array)


# Function to create a sequential keras model, involves:
# a convolution to map input image to 32 features
# a max pool to pool each 2x2 block to 1x1 block so image reduced from 32x32 to 16x16
# a dense layer to map each of the 32x16 features to a neuron
# a dropout layer to prevent over fitting
# a final dense layer to read all neurons into 2 output neurons ([1, 0] or [0, 1])
def create_model():
    model = Sequential()
    model.add(Reshape((3, 32, 32), input_shape=(3072, )))
    model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu',
                     kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    return model


# Function to train and asses model accuracy
def train_and_assess_model(model, train_images, train_labels):
    epochs = 50
    l_rate = 0.01
    decay = l_rate / epochs
    optimizer = SGD(lr=l_rate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model.summary)

    model.fit(train_images, train_labels, validation_data=(train_images, train_labels), epochs=epochs, batch_size=12)

    frozen_graph = freeze_session(K.get_session(), output_names=[model.output.op.name])
    tf.train.write_graph(frozen_graph, '.', 'face_recognition.pb', False)
    print(model.input.op.name)
    print(model.output.op.name)

    scores = model.evaluate(train_images, train_labels, verbose=0)
    print('Accuracy: %.2f%%' % (scores[1] * 100))
    return model


# Function to predict output and compare to actual output
def predict_images(model, test_images, test_labels):
    for i in range(len(test_labels)):
        test_image = np.expand_dims(test_images[i], axis=0)
        print('Predicted label: {}, actual label: {}'.format(model.predict(test_image), test_labels[i]))


# Function to get the data, format the data, train the model, and predict some test images
def run():
    face_images = load_face_images('Face_Images')
    non_face_images = load_non_face_images()

    train_images, train_labels, test_images, test_labels = separate_data(face_images, non_face_images)
    train_images = reshape_image(train_images)
    test_images = reshape_image(test_images)

    model = create_model()
    model = train_and_assess_model(model, train_images, train_labels)
    predict_images(model, test_images, test_labels)


# Run the program
run()
