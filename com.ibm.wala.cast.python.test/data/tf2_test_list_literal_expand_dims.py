"""Control for wala/ML#907: `tf.expand_dims` over a concrete scalar-list literal already resolves.

The companion witness `tf2_test_list_concat_expand_dims.py` feeds `[bos] + <opaque list>` instead;
this file pins the literal-list path so a fix for the concatenation is measured against a path
that already works.
"""

import tensorflow as tf


def f(a):
    pass


prev = tf.expand_dims([1, 2, 3], 0)
assert isinstance(prev, tf.Tensor)
assert prev.shape == (1, 3)
assert prev.dtype == tf.int32
f(prev)
