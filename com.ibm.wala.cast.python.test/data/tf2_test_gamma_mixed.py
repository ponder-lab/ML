import tensorflow as tf


def f(a):
    pass


# Test tf.random.gamma with mixed arguments
# positional: shape, alpha
# keyword: beta
alpha = tf.constant([[1.0], [3.0], [5.0]])
beta = tf.constant([[3.0, 4.0]])
samples = tf.random.gamma([30], alpha, beta=beta)

assert isinstance(samples, tf.Tensor)
# shape: [30] + broadcast(alpha.shape, beta.shape)
# alpha: [3, 1], beta: [1, 2] -> broadcast: [3, 2]
# total shape: [30, 3, 2]
assert samples.shape == (30, 3, 2)
assert samples.dtype == tf.float32

f(samples)
