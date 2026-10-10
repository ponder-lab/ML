# `tf.TensorArray`: the tensors read back from an array have the dtype it was built with.
import tensorflow as tf


def consume_stacked(x):
    assert x.dtype == tf.int32
    assert x.shape == (2,)


def consume_unstacked(x):
    assert x.dtype == tf.int32
    assert x.shape == (2,)


def consume_read(x):
    assert x.dtype == tf.int32
    assert x.shape == ()


def consume_keyword(x):
    assert x.dtype == tf.float32
    assert x.shape == (1,)


def consume_gathered(x):
    assert x.dtype == tf.int32
    assert x.shape == (1,)


written = tf.TensorArray(tf.int32, size=2)
written = written.write(0, 1)
written = written.write(1, 2)
consume_stacked(written.stack())
consume_read(written.read(0))

unstacked = tf.TensorArray(
    tf.int32, size=0, dynamic_size=True, element_shape=tf.TensorShape([])
)
unstacked = unstacked.unstack(tf.constant([3, 4]))
consume_unstacked(unstacked.stack())
consume_gathered(tf.gather_nd(unstacked.stack(), [[0]]))

floats = tf.TensorArray(dtype=tf.float32, size=1)
floats = floats.write(0, 0.5)
consume_keyword(floats.stack())
