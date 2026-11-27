# Test https://github.com/wala/ML/issues/871: a restore an exception can skip, to a handler in the
# same function, is no restore, so the setting's effect is the whole program.
import tensorflow as tf


def consume_outside(x):
    assert x.dtype == tf.float64


def risky():
    raise ValueError()


def set_float64():
    tf.keras.backend.set_floatx("float64")
    try:
        risky()
        tf.keras.backend.set_floatx("float32")
    except ValueError:
        pass


def outside():
    z = tf.keras.layers.Dense(4)(tf.ones([2, 3]))
    consume_outside(z)


set_float64()
outside()
