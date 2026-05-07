import tensorflow as tf


def f(a, b):
    pass


# Companion to `tf2_test_meshgrid.py` that exercises a 2-parameter sink
# `f(X, Y)` so the analyzer's per-parameter typing on `tf.meshgrid`'s
# tuple result can be observed at distinct value numbers.
x = tf.constant([1.0, 2.0, 3.0])
y = tf.constant([10.0, 20.0])
X, Y = tf.meshgrid(x, y)
assert isinstance(X, tf.Tensor)
assert isinstance(Y, tf.Tensor)
assert X.dtype == tf.float32
assert Y.dtype == tf.float32
f(X, Y)
