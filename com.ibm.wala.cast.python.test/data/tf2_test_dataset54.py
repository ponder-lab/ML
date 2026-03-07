import tensorflow as tf


def f(a):
    pass


def g(a):
    pass


def get_first(ds):
    my_iter = iter(ds)
    return next(my_iter)


ds1 = tf.data.Dataset.from_tensor_slices(tf.constant([1, 2, 3], dtype=tf.int32))
e1 = get_first(ds1)
assert isinstance(e1, tf.Tensor)
assert e1.shape == ()
assert e1.dtype == tf.int32
f(e1)

ds2 = tf.data.Dataset.from_tensor_slices(tf.constant([1.0, 2.0, 3.0], dtype=tf.float32))
e2 = get_first(ds2)
assert isinstance(e2, tf.Tensor)
assert e2.shape == ()
assert e2.dtype == tf.float32
g(e2)
