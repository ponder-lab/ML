import tensorflow as tf


def f(a):
    pass


ds1 = tf.data.Dataset.from_tensor_slices([[1, 2], [3, 4]])
for e1 in ds1:
    assert isinstance(e1, tf.Tensor)
    assert e1.shape == (2,)
    assert e1.dtype == tf.int32
    f(e1)
