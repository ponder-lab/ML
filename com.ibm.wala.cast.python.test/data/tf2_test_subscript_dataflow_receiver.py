# Test https://github.com/wala/ML/issues/953: a subscript of a value typed only by dataflow (a Keras
# layer's call result, reached as an element of a list parameter) takes the subscript's shape and the
# receiver's dtype, with no wholly unknown member beside it.
import tensorflow as tf


class Shift(tf.keras.layers.Layer):

    def call(self, x):
        return x + 1


def f(t):
    assert t.shape == (4,)
    assert t.dtype == tf.int32


def g(adjacency_lists):
    for i, adj in enumerate(adjacency_lists):
        f(adj[:, 0])


layer = Shift()
g([layer(tf.ones([4, 2], dtype=tf.int32))])
