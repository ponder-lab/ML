# Test https://github.com/wala/ML/issues/871: a function that restores the default on one path only
# leaves the value it set on the other, so its effect is the whole program.
import tensorflow as tf


def consume_outside(x):
    pass


def set_float64(restore):
    tf.keras.backend.set_floatx("float64")
    if restore:
        tf.keras.backend.set_floatx("float32")


def outside():
    z = tf.keras.layers.Dense(4)(tf.ones([2, 3]))
    consume_outside(z)


set_float64(False)
outside()
assert tf.keras.backend.floatx() == "float64"
