# Test https://github.com/wala/ML/issues/871: a function that sets floatx() and restores the
# default on every normal path to its exit changes it only for what runs between the two calls. A
# value the analysis cannot read, here a dtype's name, reads as the unknown dtype.
import tensorflow as tf


def consume_float64(x):
    assert x.dtype == tf.float64


def consume_unreadable(x):
    assert x.dtype == tf.float16


def consume_outside(x):
    assert x.dtype == tf.float32


def in_float64():
    tf.keras.backend.set_floatx("float64")
    y = tf.keras.layers.Dense(4)(tf.ones([2, 3]))
    consume_float64(y)
    tf.keras.backend.set_floatx("float32")


def in_unreadable(dtype):
    tf.keras.backend.set_floatx(dtype.name)
    y = tf.keras.layers.Dense(4)(tf.ones([2, 3]))
    consume_unreadable(y)
    tf.keras.backend.set_floatx("float32")


def outside():
    z = tf.keras.layers.Dense(4)(tf.ones([2, 3]))
    consume_outside(z)


in_float64()
in_unreadable(tf.float16)
outside()
