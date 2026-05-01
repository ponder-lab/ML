import tensorflow as tf


def consume(x):
    pass


a = tf.constant([1.0, 2.0, 3.0])
b = tf.constant([4.0, 5.0, 6.0])

c = a + b
assert c.shape == (3,)
assert c.dtype == tf.float32

t = (c,)

consume(t[0])
