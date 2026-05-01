import tensorflow as tf


def test(x1, x2, x3, x4):
    pass


indices = [0, 1, 2]
depth = 3

# Positional
x1 = tf.one_hot(indices, depth)
assert x1.shape.as_list() == [3, 3]
assert x1.dtype == tf.float32

# Keyword
x2 = tf.one_hot(indices=indices, depth=depth, dtype=tf.int32)
assert x2.shape.as_list() == [3, 3]
assert x2.dtype == tf.int32

# Mixed (indices pos, depth pos, axis kw)
x3 = tf.one_hot(indices, depth, axis=0)
# axis=0 -> depth dim is inserted at 0
# shape: [depth, len(indices)] = [3, 3]
assert x3.shape.as_list() == [3, 3]
assert x3.dtype == tf.float32

# Mixed (indices pos, depth pos, on_value kw, off_value kw)
x4 = tf.one_hot(indices, depth, on_value=1.0, off_value=0.0)
assert x4.shape.as_list() == [3, 3]
assert x4.dtype == tf.float32

test(x1, x2, x3, x4)
