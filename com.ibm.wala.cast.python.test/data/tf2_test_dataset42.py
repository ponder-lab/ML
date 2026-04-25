import tensorflow as tf


def f(a):
    pass


def g(a):
    pass


ds1 = tf.data.Dataset.from_tensor_slices([1, 2, 3])
for e1 in ds1:
    assert isinstance(e1, tf.Tensor)
    assert e1.shape == ()
    assert e1.dtype == tf.int32
    f(e1)

ds2 = tf.data.Dataset.from_tensor_slices([1.0, 2.0, 3.0])
for e2 in ds2:
    assert isinstance(e2, tf.Tensor)
    assert e2.shape == ()
    assert e2.dtype == tf.float32
    g(e2)
