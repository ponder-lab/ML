# Test `tf.repeat`: with an axis the output keeps the input's rank and the axis's extent is the
# total of the repeats; without one the input is flattened. A runtime tensor of repeats gives a length
# TensorFlow's static shape reports as unknown.
import tensorflow as tf


def consume_scalar(t):
    assert t.shape == (4, 3)
    assert t.dtype == tf.float32


def consume_counts(t):
    assert t.shape == (2, 6)


def consume_flat(t):
    assert t.shape == (18,)


def consume_int(t):
    assert t.shape == (2, 6)
    assert t.dtype == tf.int32


def consume_tensor(t):
    assert t.shape.as_list() == [None, 3]


x = tf.ones((2, 3))
consume_scalar(tf.repeat(x, repeats=2, axis=0))
consume_counts(tf.repeat(x, [1, 2, 3], axis=1))
consume_flat(tf.repeat(x, 3))
consume_int(tf.repeat(tf.ones((2, 3), dtype=tf.int32), 2, axis=-1))


@tf.function
def by_tensor(t, durations):
    consume_tensor(tf.repeat(t, repeats=durations, axis=0))


by_tensor(x, tf.constant([1, 2]))
