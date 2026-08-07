# The `np.random` surface of wala/ML#827. The return type is argument-dependent: with no size
# argument the draw is a Python scalar, and with one it is a `float64` array of that shape. The
# module-alias form (`rng = np.random`) is the idiom the reported subject uses.
import numpy as np
import tensorflow as tf

rng = np.random

np.random.seed(0)


def consume(w):
    pass


def consume2(a):
    pass


def consume3(b):
    pass


def consume4(c):
    pass


def consume5(d):
    pass


def consume6(e):
    pass


# A Python float initializer takes TensorFlow's weak default, `float32`, not NumPy's `float64`.
w = tf.Variable(rng.randn(), name="weight")
assert w.dtype == tf.float32 and w.shape.as_list() == []
consume(w)

a = np.random.randn(2, 3)
assert a.dtype == np.float64 and a.shape == (2, 3)
consume2(a)

b = np.random.rand(4)
assert b.dtype == np.float64 and b.shape == (4,)
consume3(b)

c = np.random.normal(0.0, 1.0, (5, 2))
assert c.dtype == np.float64 and c.shape == (5, 2)
consume4(c)

d = np.random.uniform(0.0, 1.0, (2, 2))
assert d.dtype == np.float64 and d.shape == (2, 2)
consume5(d)

# The sized family's own shapeless draw: no `size`, so a Python float rather than an array.
e = tf.Variable(np.random.normal())
assert e.dtype == tf.float32 and e.shape.as_list() == []
consume6(e)
