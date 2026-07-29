# Test https://github.com/wala/ML/issues/796: a scalar `tf.Variable` operand whose points-to set
# carries only analysis-substrate null constants must broadcast as the rank-0 identity instead of
# annihilating the elementwise composition (the `tf.matmul(x, W) + b` bias-slot regression caught
# by the 0.52.73 release smoke on TensorFlow-Examples).

import numpy as np
import tensorflow as tf

rng = np.random

W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")


def linear_regression(x):
    return W * x + b


def consume(y_pred):
    pass


X = np.array([1.0, 2.0, 3.0])
pred = linear_regression(X)
assert pred.shape == (3,)
assert pred.dtype == tf.float32
consume(pred)
