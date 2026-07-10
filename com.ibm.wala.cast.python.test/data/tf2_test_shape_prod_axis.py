import numpy as np
import tensorflow as tf


def f(a):
    pass


def get_shape(t):
    return t.shape.as_list()


# Guard fixture (wala/ML#707): `np.prod` with an extra argument (`axis=...`)
# has different semantics (the result's rank can change), so the analysis must
# refuse to fold it even though this 1-D case happens to produce the same
# scalar at runtime.
t = tf.ones((4, 5, 6))
x = tf.ones((2, 15))
r = tf.reshape(x, [np.prod(get_shape(t)[-2:], axis=0)])
assert isinstance(r, tf.Tensor)
assert r.shape == (30,)
assert r.dtype == tf.float32
f(r)
