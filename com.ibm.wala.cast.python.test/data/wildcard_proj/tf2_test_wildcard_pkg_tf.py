# Probe for wala/ML#665: `tf` reached through `from pkg.helpers2 import *` (the vendored
# `feed_forward.py` form: a package-qualified wildcard source).
from pkg.helpers2 import *


def consume(t):
    pass


x = tf.ones((4, 4))
h = tf.linalg.matmul(x, x)
consume(h)
assert h.shape == (4, 4)
