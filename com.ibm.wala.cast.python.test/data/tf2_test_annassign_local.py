import tensorflow as tf


def consume(y):
    pass


# Module-level PEP-526 annotated assignment with a value. The target `t` must be
# declared (not just assigned), or the analysis leaves it undeclared. The tensor
# value flows to `consume`.
t: tf.Tensor = tf.ones([2, 3])
assert t.shape == (2, 3) and t.dtype == tf.float32
consume(t)
