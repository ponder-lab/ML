# Probe for wala/ML#618: `tf` reached through `from helpers import *` (the vendored
# `feed_forward.py` never imports tensorflow directly; it takes `tf` from a wildcard import).
from helpers import *


def consume(t):
    pass


x = tf.ones((4, 4))
h = tf.linalg.matmul(x, x)
consume(h)
assert h.shape == (4, 4)
