# `tf.cast` of a list literal of shape elements: a vector as long as the list.
import tensorflow as tf


def consume_cast(x):
    assert x.shape == (4,)


def consume_divided(x):
    assert x.shape == (1, 4)


def consume_stacked(x):
    assert x.dtype == tf.int32
    assert x.shape == (2,)


image = tf.ones([24, 8, 3], dtype=tf.float32)
bboxes = tf.constant([[1, 1, 15, 6]], dtype=tf.float32)
scale = tf.cast(
    [image.shape[0], image.shape[1], image.shape[0], image.shape[1]], tf.float32
)
consume_cast(scale)
consume_divided(bboxes / scale)
consume_stacked(tf.stack([image.shape[0], image.shape[1]]))


def consume_nested(x):
    assert x.dtype == tf.int32
    assert x.shape == (1, 2)


consume_nested(tf.constant([[image.shape[0], image.shape[1]]]))
