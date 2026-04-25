import tensorflow as tf


def f(a):
    pass


ds1 = tf.data.Dataset.from_tensor_slices(tf.constant([1, 2, 3], dtype=tf.int32))
it1 = iter(ds1)
e1 = next(it1)
assert isinstance(e1, tf.Tensor)
assert e1.shape == ()
assert e1.dtype == tf.int32
f(e1)
