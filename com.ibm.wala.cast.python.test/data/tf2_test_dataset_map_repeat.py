import tensorflow as tf


def consume(x):
    pass


def split(record):
    a = tf.cast(record, tf.int32)
    b = tf.cast(record, tf.int64)
    return a, b


ds = tf.data.Dataset.from_tensor_slices(tf.ones([3, 4])).map(split).repeat(2)
for x, y in ds:
    assert y.shape == (4,)
    assert y.dtype == tf.int64
    consume(y)
