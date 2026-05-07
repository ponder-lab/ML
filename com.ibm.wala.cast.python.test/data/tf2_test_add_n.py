import tensorflow as tf


def f(a):
    pass


t1 = tf.constant([1, 2, 3])
t2 = tf.constant([4, 5, 6])
result = tf.add_n([t1, t2])
assert isinstance(result, tf.Tensor)
assert result.shape == (3,)
assert result.dtype == tf.int32
f(result)
