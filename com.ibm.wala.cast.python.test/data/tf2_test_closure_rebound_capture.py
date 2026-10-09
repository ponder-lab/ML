# A parameter rebound before a lambda captures it: the lambda reads the rebound value, the int32
# cast, not the float32 the parameter arrived as. Python closures bind the variable, and a closure
# created after the variable's last assignment reads that assignment alone. The loop control below
# rebinds the variable after the closure inside a loop, so a later iteration's closure call reads
# the rebound value and an earlier one the original: both bindings reach the callee.
import tensorflow as tf


def consume_captured(t):
    assert t.shape == ()
    assert t.dtype == tf.int32


def consume_loop_captured(t):
    assert t.dtype in (tf.float32, tf.int32)


def inner(image, max_area):
    consume_captured(max_area)
    return image


def erase(image, max_area=0.1):
    h, w = tf.shape(image)[-3], tf.shape(image)[-2]
    max_area = tf.cast(max_area * tf.cast(h * w, tf.float32), tf.int32)
    return tf.cond(
        tf.greater_equal(max_area, 1), lambda: inner(image, max_area), lambda: image
    )


def loop_inner(image, max_area):
    consume_loop_captured(max_area)
    return image


def loop_erase(image, max_area=0.1):
    h, w = tf.shape(image)[-3], tf.shape(image)[-2]
    for _ in range(2):
        image = tf.cond(
            tf.greater_equal(h, 1), lambda: loop_inner(image, max_area), lambda: image
        )
        max_area = tf.cast(max_area, tf.int32)
    return image


img = tf.zeros((8, 8, 3), dtype=tf.float32)
erase(img, max_area=tf.clip_by_value(tf.cast(0.5, tf.float32), 0.0, 1.0))
erase(img, max_area=0.5)
loop_erase(img, max_area=tf.clip_by_value(tf.cast(0.5, tf.float32), 0.0, 1.0))
