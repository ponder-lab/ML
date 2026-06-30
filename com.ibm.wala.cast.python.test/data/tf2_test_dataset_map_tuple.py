import tensorflow as tf


def consume(x):
    pass


def split(record):
    a = tf.cast(record, tf.int32)
    b = tf.cast(record, tf.int64)
    return a, b


# A tuple-returning map_func: the mapped element is (a, b); unpacking it yields y = b = (4,) int64
# (wala/ML#506 per-index tuple typing).
ds = tf.data.Dataset.from_tensor_slices(tf.ones([3, 4]))
ds = ds.map(split)
for x, y in ds:
    assert y.shape == (4,)
    assert y.dtype == tf.int64
    consume(y)
