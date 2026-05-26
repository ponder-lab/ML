import sys

import tensorflow as tf


def f(x):
    pass


# wala/ML#546: when `start` resolves to multiple literal values in the PA
# (here via an if/else merging two ConstantKeys into the phi), the
# cross-product of static lengths yields more than one distinct value
# (`10 - 0 = 10` vs `10 - 2 = 8`). `computeStaticInnerLength` returns
# `null` and the inner dim falls back to `RaggedDim`. Use `sys.argv` so
# the analyzer follows both branches; either runtime path satisfies the
# `r.shape == (1, None)` assertion.
if len(sys.argv) > 1:
    s = 0
else:
    s = 2
r = tf.ragged.range(s, 10, 1)
assert isinstance(r, tf.RaggedTensor)
assert r.shape == (1, None)
assert r.dtype == tf.int32
f(r)
