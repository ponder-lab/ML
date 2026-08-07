# The NumPy scalar type constructors of wala/ML#827. `np.float64` and its siblings are type
# objects, so each name has two roles: called, it builds a rank-0 array of its own dtype, and
# passed as `dtype=`, it names that dtype. Both roles are exercised here.
import numpy as np
import tensorflow as tf


def consume(a):
    pass


def consume2(b):
    pass


def consume3(c):
    pass


def consume4(d):
    pass


a = np.float64(2.0)
assert a.dtype == np.float64 and a.shape == ()
consume(a)

b = tf.Variable(np.float64(2.0))
assert b.dtype == tf.float64 and b.shape.as_list() == []
consume2(b)

c = np.int32(3)
assert c.dtype == np.int32 and c.shape == ()
consume3(c)

d = np.zeros((2, 3), dtype=np.float64)
assert d.dtype == np.float64 and d.shape == (2, 3)
consume4(d)
