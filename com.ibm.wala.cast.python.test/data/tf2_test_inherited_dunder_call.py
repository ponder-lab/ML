# Test https://github.com/wala/ML/issues/994: a Keras layer whose `__call__` is inherited from a user
# base class dispatches to that `__call__`, which delegates to the layer's `call`.
import tensorflow as tf


def consume_inherited(x):
    assert x.shape == (2, 3, 8)


def consume_inherited_call(x):
    assert x.shape == (2, 3, 8)


def consume_direct(x):
    assert x.shape == (2, 3, 8)


def consume_direct_call(x):
    assert x.shape == (2, 3, 8)


class Base(tf.keras.layers.Layer):
    def __call__(self, inputs, sequence_length=None, **kwargs):
        consume_inherited(inputs)
        return super().__call__(inputs, sequence_length=sequence_length, **kwargs)


class Inherited(Base):
    def call(self, inputs, sequence_length=None, training=None):
        consume_inherited_call(inputs)
        return inputs


class Direct(tf.keras.layers.Layer):
    def __call__(self, inputs, sequence_length=None, **kwargs):
        consume_direct(inputs)
        return super().__call__(inputs, sequence_length=sequence_length, **kwargs)

    def call(self, inputs, sequence_length=None, training=None):
        consume_direct_call(inputs)
        return inputs


x = tf.ones((2, 3, 8))
Inherited()(x, tf.constant([3, 3]), training=True)
Direct()(x, tf.constant([3, 3]), training=True)
