import tensorflow as tf


def f(a):
    pass


samples = tf.random.gamma([7, 5], [0.5, 1.5])
# samples has shape [7, 5, 2], where each slice [:, :, 0] and [:, :, 1]
# represents the 7x5 samples drawn from each of the two distributions

assert isinstance(samples, tf.Tensor)
assert samples.shape == (7, 5, 2)
assert samples.dtype == tf.float32

f(samples)
