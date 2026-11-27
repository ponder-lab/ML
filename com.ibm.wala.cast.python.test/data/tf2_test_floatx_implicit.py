# Test https://github.com/wala/ML/issues/871: a Keras dtype that defaults to floatx() is not spelled
# floatx() at all, so an unrestored set_floatx changes it too. The analysis reads the default and
# the value set, a union containing the run-time dtype.
import tensorflow as tf

tf.keras.backend.set_floatx("float64")


def consume_input(x):
    pass


def consume_dense(x):
    assert x.dtype == tf.float64


def consume_call(x):
    assert x.dtype == tf.float64


class Scale(tf.keras.layers.Layer):

    def call(self, x):
        consume_call(x)
        return x * 2.0


x_in = tf.keras.Input(shape=(7,))
assert x_in.dtype == tf.float64
consume_input(x_in)

dense = tf.keras.layers.Dense(5)(tf.ones([2, 7]))
consume_dense(dense)

Scale()(tf.ones([2, 3]))
