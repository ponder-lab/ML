# Minimal fixture mirroring `multilayer_perceptron` from
# `YunYang1994/TensorFlow2.0-Examples/2-Basical_Models/Multilayer_Perceptron.py`.
# Uses raw `tf.matmul`/`tf.add`/`tf.nn.sigmoid` instead of Keras `Dense` layers,
# with global weight/bias `tf.Variable`s — a different pattern from the
# `NeuralNet.call` cases already covered by `testNeuralNetwork*`.
import numpy as np
import tensorflow as tf

# Network parameters from the original.
n_input = 784
n_hidden_1 = 256
n_hidden_2 = 256
n_classes = 10

weights = {
    "h1": tf.Variable(tf.random.normal([n_input, n_hidden_1])),
    "h2": tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2])),
    "out": tf.Variable(tf.random.normal([n_hidden_2, n_classes])),
}
biases = {
    "b1": tf.Variable(tf.random.normal([n_hidden_1])),
    "b2": tf.Variable(tf.random.normal([n_hidden_2])),
    "out": tf.Variable(tf.random.normal([n_classes])),
}


def multilayer_perceptron(x):
    layer_1 = tf.add(tf.matmul(x, weights["h1"]), biases["b1"])
    layer_1 = tf.nn.sigmoid(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights["h2"]), biases["b2"])
    layer_2 = tf.nn.sigmoid(layer_2)
    output = tf.matmul(layer_2, weights["out"]) + biases["out"]
    return tf.nn.softmax(output)


# Driver: build a flattened mnist-style batch and pass through.
batch_x = tf.constant(np.ones((100, 784), dtype=np.float32))

assert batch_x.shape == (100, 784)
assert batch_x.dtype == tf.float32

result = multilayer_perceptron(batch_x)
assert result.shape == (100, 10)
assert result.dtype == tf.float32
