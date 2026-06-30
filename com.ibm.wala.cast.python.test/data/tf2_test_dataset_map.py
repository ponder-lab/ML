import tensorflow as tf


def consume(x):
    pass


def double(x):
    return tf.cast(x, tf.int64)


# tf.data.Dataset.map(map_func) types its elements from map_func's return (wala/ML#506): double
# casts each (4,) float32 element of the sliced dataset to (4,) int64.
ds = tf.data.Dataset.from_tensor_slices(tf.ones([3, 4]))
ds = ds.map(double)
for elem in ds:
    assert elem.shape == (4,)
    assert elem.dtype == tf.int64
    consume(elem)
