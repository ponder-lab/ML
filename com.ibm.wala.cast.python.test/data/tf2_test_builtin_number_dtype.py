import numpy as np
import tensorflow as tf


def consume_float_list(t):
    pass


def consume_float_scalar(t):
    pass


def consume_int_scalar(t):
    pass


def consume_bool_list(t):
    pass


def consume_number(t):
    pass


n = 6
s = "3"

# A Python float built by `float(...)` converts to float32.
a = tf.constant([float(n)])
assert a.shape == (1,)
assert a.dtype == tf.float32
consume_float_list(a)

b = tf.constant(float(n))
assert b.shape == ()
assert b.dtype == tf.float32
consume_float_scalar(b)

# A Python int built by `int(...)` converts to int32.
c = tf.constant(int(s))
assert c.shape == ()
assert c.dtype == tf.int32
consume_int_scalar(c)

# A Python bool built by `bool(...)` converts to bool.
d = tf.constant([bool(n)])
assert d.shape == (1,)
assert d.dtype == tf.bool
consume_bool_list(d)


# The builtins' results are Python numbers, not tensors.
x = float(n)
assert not tf.is_tensor(x)
consume_number(x)
consume_number(int(s) - 1)
consume_number(bool(n))


def consume_tf_zeros_float(t):
    pass


def consume_tf_cast_float(t):
    pass


def consume_np_float(t):
    pass


def consume_np_constant_float(t):
    pass


def consume_np_int(t):
    pass


# TensorFlow converts the Python `float` type to float32.
z = tf.zeros([2], dtype=float)
assert z.shape == (2,)
assert z.dtype == tf.float32
consume_tf_zeros_float(z)

i = tf.constant([1, 2])
k = tf.cast(i, float)
assert k.shape == (2,)
assert k.dtype == tf.float32
consume_tf_cast_float(k)

# NumPy converts a Python float to float64 and a Python int to int64.
p = np.array([float(n)])
assert p.shape == (1,)
assert p.dtype == np.float64
consume_np_float(p)

q = tf.constant(np.array([float(n)]))
assert q.shape == (1,)
assert q.dtype == tf.float64
consume_np_constant_float(q)

m = np.array([int(n)])
assert m.shape == (1,)
assert m.dtype == np.int64
consume_np_int(m)


def consume_np_full_float(t):
    pass


def consume_np_full_like_float(t):
    pass


# NumPy's conversion of a builtin-built fill value and prototype.
fl = np.full((2, 3), float(n))
assert fl.shape == (2, 3)
assert fl.dtype == np.float64
consume_np_full_float(fl)

fll = np.full_like(float(n), 3)
assert fll.shape == ()
assert fll.dtype == np.float64
consume_np_full_like_float(fll)
