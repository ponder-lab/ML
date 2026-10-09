# `tf.math.reduce_min` over a list literal of int32 scalar tensors reads int32: the random center
# from `tf.random.uniform(dtype=tf.int32)`, a difference against a `tf.shape` element, and a bound,
# in the first list, and a floor division of the bound by the first size in the second.
import tensorflow as tf


def consume_min(t):
    assert t.shape == ()
    assert t.dtype == tf.int32


def consume_min_div(t):
    assert t.shape == ()
    assert t.dtype == tf.int32


def random(max_val):
    return tf.cond(
        tf.greater(max_val, 1),
        lambda: tf.random.uniform([], 1, max_val, dtype=tf.int32),
        lambda: 1,
    )


def get_size(center1, center2, max1, max2, max_area):
    m1 = tf.math.reduce_min([center1, max1 - center1, max_area])
    consume_min(m1)
    size1 = random(m1)
    m2 = tf.math.reduce_min([center2, max2 - center2, max_area // size1])
    consume_min_div(m2)
    return size1, random(m2)


image = tf.zeros((8, 8, 3), dtype=tf.float32)
image_height, image_width = tf.shape(image)[-3], tf.shape(image)[-2]
y = tf.random.uniform([], 1, image_height - 2, dtype=tf.int32)
x = tf.random.uniform([], 1, image_width - 2, dtype=tf.int32)
max_area = tf.constant(12, dtype=tf.int32)
get_size(y, x, image_height, image_width, max_area)
