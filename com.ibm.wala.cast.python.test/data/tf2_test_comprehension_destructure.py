# Test https://github.com/wala/ML/issues/955: a comprehension whose target destructures binds every
# name it unpacks, and a comprehension or a numpy call inside a helper keeps each caller's types.
import numpy as np
import tensorflow as tf


def f(t):
    assert t.shape == (2, 3)
    assert t.dtype == tf.float32


def g(t):
    assert t.shape == (3,)
    assert t.dtype == tf.float32


def h(t):
    assert t.shape == (3,)
    assert t.dtype == tf.int32


def k(a):
    assert a.shape == (2,)
    assert a.dtype == np.int64


def m(a):
    assert a.shape == (2,)
    assert a.dtype == np.float64


class Doubler:
    # A forward-style method, which the analysis analyses per caller, so the comprehension inside
    # it can keep each caller's element type.
    def predict(self, xs):
        return [w * 2 for i, w in enumerate(xs)]


def as_arrays(xs):
    return [np.array(x) for x in xs]


pairs = [w + 1.0 for i, w in enumerate([tf.ones([2, 3]), tf.zeros([2, 3])])]
f(pairs[0])
doubler = Doubler()
g(doubler.predict([tf.ones([3])])[0])
h(doubler.predict([tf.ones([3], dtype=tf.int32)])[0])
k(as_arrays([[1, 2]])[0])
m(as_arrays([[1.0, 2.0]])[0])
