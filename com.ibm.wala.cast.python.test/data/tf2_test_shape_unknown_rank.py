import json

import tensorflow as tf


def f(a):
    pass


# The operand's shape is unresolvable to the analysis (it flows from `tf.ones(json.loads(...))`,
# as in `tf2_test_topk_unknown_input.py`), so its rank is unknown and the shape vector's extent
# must read `Unresolved`, never a concrete zero: unknown rank and rank 0 are different by contract
# (wala/ML#943). At runtime the rank is 2, so the vector is `(2,)`.
x = tf.ones(json.loads("[3, 4]"))

s = tf.shape(x)
assert isinstance(s, tf.Tensor)
assert s.shape == (2,)
assert s.dtype == tf.int32

f(s)
