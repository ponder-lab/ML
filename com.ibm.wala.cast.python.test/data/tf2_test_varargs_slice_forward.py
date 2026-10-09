# A wrapper forwards the rest of its `*args` past the first, `function(image, bboxes, *args[1:])`:
# the slice of the pack drops `args[0]`, so the bounding boxes never reach a later parameter.
from functools import wraps

import tensorflow as tf


def convert(function):
    @wraps(function)
    def wrap(image, *args, **kwargs):
        if len(args) >= 1:
            bboxes = tf.cast(args[0], tf.float32)
            image, bboxes = function(image, bboxes, *args[1:], **kwargs)
            return image, bboxes
        return function(image, **kwargs)

    return wrap


def consume_min_shape(t):
    assert t == (1, 1)


def consume_scale(t):
    assert t.dtype == tf.float32
    assert t.shape == ()


@convert
def clip(image, bboxes, min_shape=(1, 1)):
    consume_min_shape(min_shape)
    return image, bboxes


@convert
def scaled(image, bboxes, scale):
    consume_scale(scale)
    return image, bboxes


img = tf.zeros((8, 8, 3), dtype=tf.uint8)
boxes = tf.constant([[1, 1, 4, 4]], dtype=tf.int32)
clip(img, boxes)
scaled(img, boxes, tf.constant(0.5, tf.float32))
