import tensorflow as tf


def f(a):
    pass


def g(a):
    pass


def create_dataset(data):
    return tf.data.Dataset.from_tensors(data)


ds1 = create_dataset(tf.constant([1, 2], dtype=tf.int32))
for e1 in ds1:
    assert isinstance(e1, tf.Tensor)
    assert e1.shape == (2,)
    assert e1.dtype == tf.int32
    f(e1)

ds2 = create_dataset(tf.constant([[1, 2], [3, 4]], dtype=tf.int32))
for e2 in ds2:
    assert isinstance(e2, tf.Tensor)
    assert e2.shape == (2, 2)
    assert e2.dtype == tf.int32
    g(e2)
