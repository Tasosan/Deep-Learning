import tensorflow as tf
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib

# Saving the graph as a pb file taking data from pbtxt and ckpt files and providing a few operations
freeze_graph.freeze_graph('advanced_mnist.pbtxt',
                          '',
                          True,
                          'advanced_mnist.ckpt',
                          'y_readout1',
                          'save/restore_all',
                          'save/Const:0',
                          'frozen_advanced_mnist.pb',
                          True,
                          '')

# Read the data form the frozen graph pb file
input_graph_def = tf.GraphDef()
with tf.gfile.Open('frozen_advanced_mnist.pb', 'rb') as f:
    data = f.read()
    input_graph_def.ParseFromString(data)

# Optimize the graph with input and output nodes
output_graph_def = optimize_for_inference_lib.optimize_for_inference(
    input_graph_def,
    ['x_input', 'keep_prob'],
    ['y_readout1'],
    tf.float32.as_datatype_enum)

# Save the optimized graph to the optimized pb file
f = tf.gfile.FastGFile('optimized_advanced_mnist.pb', 'w')
f.write(output_graph_def.SerializeToString())
