import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)


# Function to create a weight neuron using a random number. Training will assign a real weight later
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


# Function to create a bias neuron. Bias of 0.1 will help to prevent any 1 neuron from being chosen too often
def biases_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


# Function to create a convolutional neuron. Convolutes input from 4d to 2d. This helps streamline inputs
def conv_2d(x, W, name):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)


# Function to create a neuron to represent the max input. Helps to make the best prediction for what comes next
def max_pool(x, name):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


# A way to input images (as 784 element arrays of pixel values 0 - 1)
x_input = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x_input')
# A way to input labels to show model what the correct answer is during training
y_input = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y_input')

# First convolutional layer - reshape/resize images
# A weight variable that examines batches of 5x5 pixels, returns 32 features (1 feature per bit value in 32 bit float)
W_conv1 = weight_variable([5, 5, 1, 32], 'W_conv1')
# Bias variable to add to each of the 32 features
b_conv1 = biases_variable([32], 'b_conv1')
# Reshape each input image into a 28 x 28 x 1 pixel matrix
x_image = tf.reshape(x_input, [-1, 28, 28, 1], name='x_image')
# Flattens filter (W_conv1) to [5 * 5 * 1, 32], multiplies by [None, 28, 28, 1] to associate each 5x5 batch with the
# 32 features, and adds biases
h_conv1 = tf.nn.relu(conv_2d(x_image, W_conv1, name='conv1') + b_conv1, name='h_conv1')
# Takes windows of size 2x2 and computes a reduction on the output of h_conv1 (computes max, used for better prediction)
# Images are reduced to size 14 x 14 for analysis
h_pool1 = max_pool(h_conv1, name='h_pool1')

# Second convolutional layer, reshape/resize images
# Does mostly the same as above but converts each 32 unit output tensor from layer 1 to a 64 feature tensor
W_conv2 = weight_variable([5, 5, 32, 64], 'W_conv2')
b_conv2 = biases_variable([64], 'b_conv2')
h_conv2 = tf.nn.relu(conv_2d(h_pool1, W_conv2, name='conv2') + b_conv2, name='h_conv2')
# Images at this point are reduced to size 7 x 7 for analysis
h_pool2 = max_pool(h_conv2, name='h_pool2')

# First dense layer, performing calculation based on previous layer output
# Each image is 7 x 7 at the end of the previous section and outputs 64 features, we want 32 x 32 neurons = 1024
W_dense1 = weight_variable([7 * 7 * 64, 1024], name='W_dense1')
# bias variable added to each output feature
b_dense1 = biases_variable([1024], name='b_dense1')
# Flatten each of the images into size [None, 7 x 7 x 64]
h_pool_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64], name='h_pool_flat')
# Multiply weights by the outputs of the flatten neuron and add biases
h_dense1 = tf.nn.relu(tf.matmul(h_pool_flat, W_dense1, name='matmul_dense1') + b_dense1, name='h_dense1')

# Dropout layer prevents overfitting or recognizing patterns where none exist
# Depending on what value we enter into keep_prob, it will apply or not apply dropout layer
keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
# Dropout layer will be applied during training but not testing or predicting
h_drop1 = tf.nn.dropout(h_dense1, keep_prob, name='h_drop1')

# Readout layer used to format output
# Weight variable takes inputs from each of the 1024 neurons from before and outputs an array of 10 elements
W_readout1 = weight_variable([1024, 10], name='W_readout1')
# Apply bias to each of the 10 outputs
b_readout1 = biases_variable([10], name='b_readout1')
# Perform final calculation by multiplying each of the neurons from dropout layer by weights and adding biases
y_readout1 = tf.add(tf.matmul(h_drop1, W_readout1, name='matmul_readout1'), b_readout1, name='y_readout1')

# Softmax cross entropy loss function compares expected answers (labels) vs actual answers (logits)
cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=y_readout1))
# Adam optimizer aims to minimize loss
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy_loss)
# Compare actual vs expected outputs to see if highest number is at the same index, true if they match and false if not
correct_prediction = tf.equal(tf.argmax(y_input, 1), tf.argmax(y_readout1, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Used to save the graph and weights
saver = tf.train.Saver()

# Run in with statement so session only exists within it
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Save the graph shape and node names to pbtxt file
    tf.train.write_graph(sess.graph_def, '.', 'advanced_mnist.pbtxt', False)

    # Train the model, running through data 20000 times in batches of 50
    # Print out step # and accuracy every 100 steps and final accuracy at the end of training
    # Train by running train_step and apply dropout by setting keep_prob to 0.5
    for i in range(20000):
        batch = mnist_data.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x_input: batch[0], y_input: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" %(i, train_accuracy))
        train_step.run(feed_dict={x_input: batch[0], y_input: batch[1], keep_prob: 0.5})
    print("test accuracy %g" % accuracy.eval(feed_dict={x_input: mnist_data.test.images,
                                                        y_input: mnist_data.test.labels, keep_prob: 1.0}))

    # Save the session with graph shape and node weights
    saver.save(sess, 'advanced_mnist.ckpt')

    # Make a prediction
    print(sess.run(y_readout1, feed_dict={x_input: [mnist_data.test.images[0]], keep_prob: 1.0}))