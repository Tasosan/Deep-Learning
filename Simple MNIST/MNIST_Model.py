import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Contains all of the images and labels (train and test) in the MNIST_data data set
mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

# train_image = mnist_data.train.images[0]
# train_label = mnist_data.train.labels[0]
# print(train_image)
# print(train_label)

# y = Wx + b
# Input to the graph, takes in any number of images (784 element pixel arrays)
x_input = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x_input')
# Weights to be multiplied by input
W = tf.Variable(initial_value=tf.zeros(shape=[784, 10]), name='W')
# Biases to be added to weights * inputs
b = tf.Variable(initial_value=tf.zeros(shape=[10]), name='b')
# Actual model prediction based on input and current values of W and b
y_actual = tf.add(x=tf.matmul(a=x_input, b=W, name='matmul'), y=b, name='y_actual')
# Input to enter correct answer for comparison during training
y_expected = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y_expected')

# Cross entropy loss function because output is a list of possibilities (% certainty of the correct answer)
cross_entropy_loss = tf.reduce_mean(
    input_tensor=tf.nn.softmax_cross_entropy_with_logits(labels=y_expected, logits=y_actual),
    name='cross_entropy_loss')
# Classic gradient descent optimizer aims to minimize the difference between expected and actual values (loss)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5, name='optimizer')
train_step = optimizer.minimize(loss=cross_entropy_loss, name='train_step')

saver = tf.train.Saver()

# Create the session to run the nodes
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

tf.train.write_graph(graph_or_graph_def=session.graph_def,
                     logdir='.',
                     name='mnist_model.pbtxt',
                     as_text=False)

# Train the model by fetching batches of 100 images and labels at a time and running train_step
# Run through the batches 1000 times (epochs)
for _ in range(1000):
    batch = mnist_data.train.next_batch(100)
    train_step.run(feed_dict={x_input: batch[0], y_expected: batch[1]})

saver.save(sess=session,
           save_path='mnist_model.ckpt')

# Measure accuracy by comparing the predicted values to the correct values and calculating how many of them match
correct_prediction = tf.equal(x=tf.argmax(y_actual, 1), y=tf.argmax(y_expected, 1))
accuracy = tf.reduce_mean(tf.cast(x=correct_prediction, dtype=tf.float32))
print(accuracy.eval(feed_dict={x_input: mnist_data.test.images, y_expected: mnist_data.test.labels}))

# Test a prediction on a single image
print(session.run(fetches=y_actual, feed_dict={x_input: [mnist_data.test.images[0]]}))
