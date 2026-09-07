# Witness for wala/ML#878: matmul broadcasts its BATCH axes. Taking the higher-rank operand's
# prefix gets this wrong on two axes at equal rank, which is the case the old comment called the
# sound one: (3, 1, 2, 4) against (1, 5, 4, 6) has both operands at rank 4, so the tie went to the
# first operand and produced (3, 1, 2, 6) where TensorFlow produces (3, 5, 2, 6).
import tensorflow as tf


def consume(x):
    pass


def consume_unequal_rank(x):
    pass


a = tf.zeros((3, 1, 2, 4))
b = tf.zeros((1, 5, 4, 6))
c = tf.matmul(a, b)
consume(c)
assert c.shape == (3, 5, 2, 6)

# The unequal-rank form of the same defect: the shorter operand's batch axis has to widen the
# longer one's `1` rather than being dropped.
d = tf.zeros((1, 2, 4))
e = tf.zeros((7, 4, 6))
f = tf.matmul(d, e)
consume_unequal_rank(f)
assert f.shape == (7, 2, 6)
