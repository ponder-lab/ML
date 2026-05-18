# Minimal fixture for `tf2_test_multilayer_perceptron.py`'s companion
# function. Mirrors `accuracy` from
# `YunYang1994/TensorFlow2.0-Examples/2-Basical_Models/Multilayer_Perceptron.py`.
# Distinct from `neural_network.py`'s `accuracy` (Dense-layer-chain variant
# from a different repo, covered by `testNeuralNetwork4`); this is the
# raw-`tf.matmul` MLP companion.
import tensorflow as tf


def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


# Driver: shapes mirror `multilayer_perceptron`'s output (batch, num_classes).
y_pred = tf.constant([[0.1, 0.9], [0.7, 0.3]], dtype=tf.float32)
y_true = tf.constant([1, 0], dtype=tf.int64)

assert y_pred.shape == (2, 2)
assert y_pred.dtype == tf.float32
assert y_true.shape == (2,)
assert y_true.dtype == tf.int64

accuracy(y_pred, y_true)
