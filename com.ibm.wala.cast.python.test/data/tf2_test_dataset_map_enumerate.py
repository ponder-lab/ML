import tensorflow as tf


def consume(x):
    pass


def split(record):
    a = tf.cast(record, tf.int32)
    b = tf.cast(record, tf.int64)
    return a, b


# The gpt-2 fit-loop shape (wala/ML#618): enumerate over a mapped tuple dataset with nested
# unpacking. y = b = (4,) int64 (wala/ML#506).
ds = tf.data.Dataset.from_tensor_slices(tf.ones([3, 4]))
ds = ds.map(split)
for i, (x, y) in enumerate(ds):
    assert y.shape == (4,)
    assert y.dtype == tf.int64
    consume(y)
