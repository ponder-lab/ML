import numpy as np
import tensorflow as tf


def consume_raw(x):
    pass


def consume_div(x):
    pass


def consume_axis(x):
    pass


def consume_astype_div(x):
    pass


def consume_astype_axis(x):
    pass


a = np.ones((96, 28, 28), dtype=np.uint8)
consume_raw(a)

b = a / 255.0
assert b.shape == (96, 28, 28) and b.dtype == np.float64
consume_div(b)

c = b[..., tf.newaxis]
assert c.shape == (96, 28, 28, 1)
consume_axis(c)

d = a.astype(np.float32) / 255.0
assert d.shape == (96, 28, 28) and d.dtype == np.float32
consume_astype_div(d)

e = d[..., tf.newaxis]
assert e.shape == (96, 28, 28, 1)
consume_astype_axis(e)


def consume_destructured(x):
    pass


f, g = a / 255.0, a / 128.0
assert f.shape == (96, 28, 28) and f.dtype == np.float64
consume_destructured(f)
