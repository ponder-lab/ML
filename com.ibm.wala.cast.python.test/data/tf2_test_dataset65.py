import tensorflow as tf


def f(a):
    pass


def create_dataset(start, stop):
    return tf.data.Dataset.range(start, stop)


ds1 = create_dataset(1, 5)
for e1 in ds1:
    assert isinstance(e1, tf.Tensor)
    assert e1.shape == ()
    assert e1.dtype == tf.int64
    f(e1)
