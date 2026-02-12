import tensorflow as tf


def f(a):
    pass


a = [0.5, 1.5]
assert isinstance(a, list)
assert len(a) == 2
assert all(isinstance(x, float) for x in a)
assert tf.shape(a) == (2,)

samples = tf.random.gamma([10], a, None, tf.float64)
# samples has shape [10, 2], where each slice [:, 0] and [:, 1] represents
# the samples drawn from each distribution
assert isinstance(samples, tf.Tensor)
assert samples.shape == (10, 2)
assert samples.dtype == tf.float64

f(samples)
