# One converting decorator over two functions of different arity; the keyword parameter of the
# second must not receive the first's bounding boxes. No lambda capture.
import tensorflow as tf
from functools import wraps


def convert(function):
    @wraps(function)
    def wrap(image, *args, **kwargs):
        image = tf.image.convert_image_dtype(image, tf.float32)
        if len(args) >= 1:
            bboxes = tf.cast(args[0], tf.float32)
            image, bboxes = function(image, bboxes, *args[1:], **kwargs)
            return image, bboxes
        image = function(image, **kwargs)
        return image

    return wrap


def consume_param(t):
    assert t.dtype == tf.float32
    assert t.shape == ()


def consume_boxes(t):
    assert t.dtype == tf.float32
    assert t.shape == (1, 4)


@convert
def with_boxes(image, bboxes, min_shape=(1, 1)):
    consume_boxes(bboxes)
    return image, bboxes


@convert
def erase(image, max_area=0.1, erased_value=0):
    consume_param(max_area)
    return image


img = tf.zeros((8, 8, 3), dtype=tf.uint8)
boxes = tf.constant([[1, 1, 4, 4]], dtype=tf.int32)
with_boxes(img, boxes)
erase(img, max_area=tf.constant(0.5, tf.float32))
