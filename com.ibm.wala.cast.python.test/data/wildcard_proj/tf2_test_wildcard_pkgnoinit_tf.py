# Probe for wala/ML#684: `tf` reached through `from pkgnoinit.helpers3 import *` — a
# package-qualified wildcard source in a package WITHOUT `__init__.py`, whose module also reads
# `tf` in one of its own functions. This is the exact `from custom.layers import *` shape of
# MusicTransformer-tensorflow2.0.
from pkgnoinit.helpers3 import *


def consume(t):
    pass


x = tf.ones((4, 4))
consume(x)
assert x.shape == (4, 4)
