import tensorflow as tf


def f(a):
    pass


def create_dataset(data):
    return tf.data.Dataset.from_tensors(data)


ds1 = create_dataset(tf.constant([1, 2, 3], dtype=tf.int32))
for e1 in ds1:
    assert isinstance(e1, tf.Tensor)
    assert e1.shape == (3,)
    assert e1.dtype == tf.int32
    f(e1)
