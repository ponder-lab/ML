# Test for wala/ML#1108: applying a program class that subclasses a built-in Keras layer runs the
# built-in layer's `call`, whether the class inherits it or overrides `call` and calls
# `super().call`, so the application has the type the built-in layer gives its input.
import tensorflow as tf


class Frozen(tf.keras.layers.BatchNormalization):

    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


class Plain(tf.keras.layers.BatchNormalization):
    pass


class Wider(tf.keras.layers.Dense):
    pass


class Sized(tf.keras.layers.Dense):

    def __init__(self, n):
        super().__init__(n)


def consume_frozen(x):
    assert x.shape == (2, 3) and x.dtype == tf.float32
    return x


def consume_plain(x):
    assert x.shape == (2, 3) and x.dtype == tf.float32
    return x


def consume_wider(x):
    assert x.shape == (2, 4) and x.dtype == tf.float32
    return x


def consume_keyword(x):
    assert x.shape == (2, 5) and x.dtype == tf.float32
    return x


def make(cls, n):
    return cls(n)


def consume_made_six(x):
    assert x.shape == (2, 6) and x.dtype == tf.float32
    return x


def consume_made_seven(x):
    assert x.shape == (2, 7) and x.dtype == tf.float32
    return x


def consume_sized(x):
    assert x.shape == (2, 4) and x.dtype == tf.float32
    return x


consume_frozen(Frozen()(tf.random.normal([2, 3])))
consume_plain(Plain()(tf.random.normal([2, 3])))
consume_wider(Wider(4)(tf.random.normal([2, 3])))
consume_keyword(Wider(use_bias=False, units=5)(tf.random.normal([2, 3])))
consume_made_six(make(Wider, 6)(tf.random.normal([2, 3])))
consume_made_seven(make(Wider, 7)(tf.random.normal([2, 3])))
consume_sized(Sized(4)(tf.random.normal([2, 3])))
