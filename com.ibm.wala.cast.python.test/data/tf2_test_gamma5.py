import tensorflow as tf


def f(a):
    pass


alpha = tf.constant([[1.0], [3.0], [5.0]])
beta = tf.constant([[3.0, 4.0]])
samples = tf.random.gamma([30], alpha, beta)
# samples has shape [30, 3, 2], with 30 samples each of 3x2 distributions.

assert isinstance(samples, tf.Tensor)
assert samples.shape == (30, 3, 2)
assert samples.dtype == tf.float32

f(samples)
