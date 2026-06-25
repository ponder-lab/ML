import tensorflow as tf


def consume_transpose_default(a):
    pass


def consume_transpose_perm(a):
    pass


def consume_transpose_tensor_perm(a):
    pass


# Default transpose reverses the axes: (2, 3) -> (3, 2).
t = tf.transpose(tf.ones((2, 3)))
assert isinstance(t, tf.Tensor)
assert t.shape == (3, 2)
assert t.dtype == tf.float32
consume_transpose_default(t)

# An explicit perm permutes the axes: (2, 3, 4) with perm [0, 2, 1] -> (2, 4, 3).
tp = tf.transpose(tf.ones((2, 3, 4)), perm=[0, 2, 1])
assert isinstance(tp, tf.Tensor)
assert tp.shape == (2, 4, 3)
assert tp.dtype == tf.float32
consume_transpose_perm(tp)

# A perm passed as a tensor constant (rather than a Python list literal) is
# still resolved to its constant values, so the permuted shape is recovered
# precisely: (2, 3, 4) with perm [2, 1, 0] -> (4, 3, 2).
dyn = tf.transpose(tf.ones((2, 3, 4)), tf.constant([2, 1, 0]))
assert isinstance(dyn, tf.Tensor)
assert dyn.shape == (4, 3, 2)
assert dyn.dtype == tf.float32
consume_transpose_tensor_perm(dyn)
