# The deterministic `tf.image` flips (wala/ML#1262): `flip_left_right` and `flip_up_down` each
# return a tensor of the same shape and dtype as their image argument, whether the image is
# passed by position or by keyword, a single image or a batch, and through a chain of flips.
import tensorflow as tf


def consume_left_right(x):
    assert x.shape == (2, 3, 3)
    assert x.dtype == tf.float32


def consume_left_right_keyword(x):
    assert x.shape == (2, 3, 3)
    assert x.dtype == tf.float32


def consume_up_down(x):
    assert x.shape == (4, 2, 3, 1)
    assert x.dtype == tf.uint8


def consume_up_down_keyword(x):
    assert x.shape == (4, 2, 3, 1)
    assert x.dtype == tf.uint8


def consume_chain(x):
    assert x.shape == (2, 3, 3)
    assert x.dtype == tf.float32


c = tf.ones([2, 3, 3])
consume_left_right(tf.image.flip_left_right(c))
consume_left_right_keyword(tf.image.flip_left_right(image=c))

batch = tf.zeros((4, 2, 3, 1), dtype=tf.uint8)
consume_up_down(tf.image.flip_up_down(batch))
consume_up_down_keyword(tf.image.flip_up_down(image=batch))

consume_chain(tf.image.flip_up_down(tf.image.flip_left_right(c)))
