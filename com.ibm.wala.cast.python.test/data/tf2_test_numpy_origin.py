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


def np_add3(a, b, c):
    # A nested binary operator: the outer operand is itself a binop result with no points-to set,
    # exercising the structural recursion in the origin classification.
    return a + b + c


def scale(m):
    # A statically opaque scalar co-operand keeps the tensor operand's library.
    return m * 2.0


def make_opaque(k):
    # The shape argument is a non-constant expression, so the analysis cannot resolve the array's
    # shape (an unknown tensor, ⊤); only the numpy origin is known. Exercises origin propagation
    # through a null-state predecessor.
    return np.zeros((k * 1,), dtype=np.float32)


def consume_np(x):
    pass


def consume_np_opaque(x):
    pass


def consume_np_loader(x):
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

s3 = np_add3(m, m, m)
assert isinstance(s3, np.ndarray)
assert s3.shape == (3,)

sc = scale(m)
assert isinstance(sc, np.ndarray)
assert sc.shape == (3,)

# Keras dataset loaders return plain ndarrays at runtime, so their results are numpy-origin (the
# consumer-ratified runtime-type rule).
(mn_x, mn_y), _ = tf.keras.datasets.mnist.load_data()
assert isinstance(mn_x, np.ndarray)
(c10_x, c10_y), _ = tf.keras.datasets.cifar10.load_data()
assert isinstance(c10_x, np.ndarray)
(c100_x, c100_y), _ = tf.keras.datasets.cifar100.load_data()
assert isinstance(c100_x, np.ndarray)
(im_x, im_y), _ = tf.keras.datasets.imdb.load_data()
assert isinstance(im_x, np.ndarray)
(re_x, re_y), _ = tf.keras.datasets.reuters.load_data()
assert isinstance(re_x, np.ndarray)
(bh_x, bh_y), _ = tf.keras.datasets.boston_housing.load_data()
assert isinstance(bh_x, np.ndarray)

consume_np(y)
consume_np(r)
consume_np(z)
consume_np(s3)
consume_np(sc)
consume_np_opaque(o)
consume_np_loader(mn_x)
consume_np_loader(c10_x)
consume_np_loader(c100_x)
consume_np_loader(im_x)
consume_np_loader(re_x)
consume_np_loader(bh_x)
consume_tf(w)
