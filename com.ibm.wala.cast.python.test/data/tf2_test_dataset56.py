import tensorflow as tf


def f(a):
    pass


def g(a):
    pass


def process_dataset(ds):
    return ds.shuffle(10)


ds1 = tf.data.Dataset.from_tensor_slices(tf.constant([1, 2, 3], dtype=tf.int32))
shuffled_ds1 = process_dataset(ds1)
for e1 in shuffled_ds1:
    assert isinstance(e1, tf.Tensor)
    assert e1.shape == ()
    assert e1.dtype == tf.int32
    f(e1)

ds2 = tf.data.Dataset.from_tensor_slices(tf.constant([1.0, 2.0, 3.0], dtype=tf.float32))
shuffled_ds2 = process_dataset(ds2)
for e2 in shuffled_ds2:
    assert isinstance(e2, tf.Tensor)
    assert e2.shape == ()
    assert e2.dtype == tf.float32
    g(e2)
