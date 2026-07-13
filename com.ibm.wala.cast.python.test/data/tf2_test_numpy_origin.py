# Test numpy-origin vs TensorFlow-origin tensor values (wala/ML#724). Each of the three
# instruction shapes from the issue produces a tensor-typed value whose producing library the
# analysis should record: a binary operator over numpy operands, an ndarray method call, and an
# interprocedural return of `np.array(...)`. The mixed binary operator dispatches to TensorFlow
# at runtime and therefore counts as TensorFlow-origin.
import numpy as np
import tensorflow as tf


def np_add(a, b):
    return a + b


def np_method(m):
    return m.reshape((1, 3))


def make_array():
    return np.array([1.0, 2.0, 3.0])


def mixed(a, t):
    return a + t


def make_opaque(k):
    # The shape argument is a non-constant expression, so the analysis cannot resolve the array's
    # shape (an unknown tensor, ⊤); only the numpy origin is known. Exercises origin propagation
    # through a null-state predecessor.
    return np.zeros((k * 1,), dtype=np.float32)


def consume_np(x):
    pass


def consume_np_opaque(x):
    pass


def consume_tf(x):
    pass


m = np.zeros((3,), dtype=np.float32)

y = np_add(m, m)
assert isinstance(y, np.ndarray)
assert y.shape == (3,)

r = np_method(m)
assert isinstance(r, np.ndarray)
assert r.shape == (1, 3)

z = make_array()
assert isinstance(z, np.ndarray)
assert z.shape == (3,)

t = tf.constant([1.0, 2.0, 3.0])

w = mixed(m, t)
assert isinstance(w, tf.Tensor)
assert w.shape == (3,)
assert w.dtype == tf.float32

o = make_opaque(4)
assert isinstance(o, np.ndarray)
assert o.shape == (4,)

consume_np(y)
consume_np(r)
consume_np(z)
consume_np_opaque(o)
consume_tf(w)
