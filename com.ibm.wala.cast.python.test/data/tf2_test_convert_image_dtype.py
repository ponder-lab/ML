# `tf.image.convert_image_dtype` returns the image's shape in the requested dtype.
import tensorflow as tf


def consume_float(x):
    assert x.dtype == tf.float32
    assert x.shape == (8, 8, 3)


def consume_back(x):
    assert x.dtype == tf.uint8
    assert x.shape == (8, 8, 3)


image = tf.zeros((8, 8, 3), dtype=tf.uint8)
converted = tf.image.convert_image_dtype(image, tf.float32)
consume_float(converted)
consume_back(tf.image.convert_image_dtype(converted, tf.uint8, saturate=True))
