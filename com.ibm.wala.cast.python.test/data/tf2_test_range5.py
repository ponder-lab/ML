import tensorflow as tf


def f(x):
    pass


r = tf.range(12)
r2 = tf.reshape(r, [3, 4])

for i in r2:
    assert i.shape == (4,)
    assert i.dtype == tf.int32
    f(i)
