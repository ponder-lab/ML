import tensorflow as tf


def f(a):
    pass


samples = tf.random.poisson([10], [0.5, 1.5], tf.float64)
# samples has shape [10, 2], where each slice [:, 0] and [:, 1] represents
# the samples drawn from each distribution
assert samples.shape == (10, 2)
assert samples.dtype == tf.float32

f(samples)
