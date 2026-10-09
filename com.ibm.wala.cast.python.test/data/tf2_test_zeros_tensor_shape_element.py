# Test an allocator whose shape list holds a scalar tensor: the list's length fixes the rank, and the
# tensor element is a dimension TensorFlow's static shape reports as unknown.
import tensorflow as tf


def consume_reduced(t):
    assert t.shape.as_list() == [0, None, 8]


def consume_ones(t):
    assert t.shape.as_list() == [3, None]


def consume_either(t):
    assert t.shape.as_list() in ([4, 2], [None, 2])


@tf.function
def build(durations, static=True):
    max_durations = tf.reduce_max(durations)
    consume_reduced(tf.zeros(shape=[0, max_durations, 8], dtype=tf.float32))
    consume_ones(tf.ones([3, max_durations]))
    n = 4 if static else max_durations
    consume_either(tf.zeros([n, 2]))


build(tf.constant([2, 4, 1]))
