# Test https://github.com/wala/ML/issues/947: a value typed only by dataflow (a Keras layer's call
# result) keeps its type through `tf.convert_to_tensor` and through a summary routed through it.
import tensorflow as tf


class Scale(tf.keras.layers.Layer):

    def call(self, x):
        return x * 2.0


def f(t):
    assert t.shape == (2, 3)
    assert t.dtype == tf.float32


def g(t):
    assert t.shape == (2, 3)
    assert t.dtype == tf.float32


layer = Scale()
f(tf.tanh(layer(tf.ones([2, 3]))))
g(tf.convert_to_tensor(layer(tf.ones([2, 3]))))
