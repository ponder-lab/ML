import tensorflow as tf
import numpy as np

# Mirrors `logistic_regression` from
# `TensorFlow-Examples/.../2_BasicModels/logistic_regression.py`, a real-world
# image-classification utility (logistic regression: `softmax(W x + b)` over
# module-level weight/bias variables), for tensor-type inference coverage.
num_features = 784
num_classes = 10

W = tf.Variable(tf.ones([num_features, num_classes]), name="weight")
b = tf.Variable(tf.zeros([num_classes]), name="bias")


def logistic_regression(x):
    # Apply softmax to normalize the logits to a probability distribution.
    return tf.nn.softmax(tf.matmul(x, W) + b)


x = tf.constant(np.ones((100, num_features), dtype=np.float32))
result = logistic_regression(x)
assert x.shape == (100, num_features) and x.dtype == tf.float32
assert result.shape == (100, num_classes) and result.dtype == tf.float32
