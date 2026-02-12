import tensorflow as tf


@tf.function(reduce_retracing=True)
def returned(a):
    return a


a = tf.range(5)

assert a.shape == (5,)
assert a.dtype == tf.int32

b = returned(a)
