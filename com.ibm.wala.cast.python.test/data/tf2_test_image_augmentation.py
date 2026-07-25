# The shape-and-dtype-preserving `tf.image` augmentation ops (wala/ML#792), mirroring the
# corpus augmentation chain: each op returns a tensor of the same shape and dtype as its
# image argument, and the dtype must also survive a hop whose shape is honestly unknown
# (`tf.slice` with runtime extents).
import tensorflow as tf


def consume(x):
    pass


def consume2(x):
    pass


img = tf.zeros((4, 5, 3), dtype=tf.uint8)

a = tf.image.adjust_brightness(img, delta=0.1)
a = tf.image.random_flip_left_right(a)
a = tf.image.adjust_contrast(a, contrast_factor=1.2)
a = tf.image.adjust_saturation(a, saturation_factor=1.1)
a = tf.image.adjust_hue(a, delta=0.02)
assert a.dtype == tf.uint8
assert a.shape == (4, 5, 3)
consume(a)

begin = tf.constant([0, 0, 0])
size = tf.constant([2, 2, 3])
cropped = tf.slice(img, begin, size)
b = tf.image.adjust_brightness(cropped, delta=0.1)
assert b.dtype == tf.uint8
consume2(b)
