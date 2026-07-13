# Test the first-class parameter origin (wala/ML#726). A tensor parameter is the hybridization
# frame's symbolic value: under `tf.function` tracing it is a tensor regardless of the library
# that produced its eager feeds, so a parameter reads PARAMETER (not its call sites' origins), an
# operator over a parameter stays PARAMETER, and locally-derived numpy values keep reading NUMPY.
import numpy as np
import tensorflow as tf


def f(x):
    # The wala/ML#726 distillation: fed an ndarray, `x + 1` still lowers to `tf.add` under
    # tracing, so the result carries the parameter origin, not numpy.
    return x + 1


def numpy_body():
    # The counter-case: numpy literals, a binary operator over locals, and an interprocedural
    # numpy return, with no parameter provenance in any def; pure numpy origin must survive the
    # parameter machinery.
    m = np.zeros((2, 3), dtype=np.float32)
    y = m + m
    return np.array(y)


def consume_param(x):
    pass


def consume_np(x):
    pass


def consume_tf(x):
    pass


a = f(np.ones((2, 3)))
assert isinstance(a, np.ndarray)
assert a.shape == (2, 3)

b = numpy_body()
assert isinstance(b, np.ndarray)
assert b.shape == (2, 3)
assert b.dtype == np.float32

# A mixed operator over locals pins the runtime-dispatch rule below the parameter boundary:
# `ndarray + Tensor` dispatches to TensorFlow and yields a `tf.Tensor`.
m = np.zeros((3,), dtype=np.float32)
t = tf.constant([1.0, 2.0, 3.0])
w = m + t
assert isinstance(w, tf.Tensor)
assert w.shape == (3,)
assert w.dtype == tf.float32

consume_param(a)
consume_np(b)
consume_tf(w)
