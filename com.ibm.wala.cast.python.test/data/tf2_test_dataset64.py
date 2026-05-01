import tensorflow as tf


def f(a):
    pass


def g(a):
    pass


def create_dataset(start, stop):
    # Output dtype of range is int64
    return tf.data.Dataset.range(start, stop)


ds1 = create_dataset(1, 5)  # Generates 1, 2, 3, 4
for e1 in ds1:
    assert isinstance(e1, tf.Tensor)
    assert e1.shape == ()
    assert e1.dtype == tf.int64
    f(e1)

ds2 = create_dataset(10, 12)  # Generates 10, 11
for e2 in ds2:
    assert isinstance(e2, tf.Tensor)
    assert e2.shape == ()
    assert e2.dtype == tf.int64
    g(e2)
