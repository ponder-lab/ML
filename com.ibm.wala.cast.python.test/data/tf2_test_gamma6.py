import tensorflow as tf


def f(a):
    pass


alpha = [[1.0], [3.0], [5.0]]
assert isinstance(alpha, list)
assert len(alpha) == 3
assert tf.constant(alpha).shape == (3, 1)

beta = [[3.0, 4.0]]
assert isinstance(beta, list)
assert len(beta) == 1
assert tf.constant(beta).shape == (1, 2)

res = tf.constant(alpha) + tf.constant(beta)
assert res.shape == (3, 2)


samples = tf.random.gamma([30], alpha, beta)
# samples has shape [30, 3, 2], with 30 samples each of 3x2 distributions.

assert isinstance(samples, tf.Tensor)
assert samples.shape == (30, 3, 2)
assert samples.dtype == tf.float32

f(samples)
