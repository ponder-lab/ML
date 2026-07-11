import tensorflow as tf


def consume(t):
    pass


# NLPGNN's ALBERT entry idiom in miniature (wala/ML#717): a stacked input is
# split three ways on axis 0 and the pieces unpack; each piece carries the
# quotient shape.
x = tf.ones((3, 8, 100))
a, b, c = tf.split(x, 3, 0)

assert a.shape == (1, 8, 100)
assert b.shape == (1, 8, 100)
assert c.shape == (1, 8, 100)
assert a.dtype == tf.float32

consume(a)
consume(c)


def consume_default_axis(t):
    pass


def consume_size_list(t):
    pass


# The axis defaults to 0 when absent; a size-list split produces
# differently-shaped pieces, which the single-piece model soundly
# represents with a dynamic dimension at the axis.
x2 = tf.ones((4, 6))
p, q = tf.split(x2, 2)
r, s = tf.split(x2, [1, 3], 0)

assert p.shape == (2, 6)
assert q.shape == (2, 6)
assert r.shape == (1, 6)
assert s.shape == (3, 6)

consume_default_axis(p)
consume_size_list(s)


def consume_opaque_axis(t):
    pass


# A non-constant axis leaves the output shape unknown, but the dtype still
# inherits from the value.
k = x2.shape.ndims - 2
u, v = tf.split(x2, 2, k)

assert u.shape == (2, 6)
assert v.shape == (2, 6)

consume_opaque_axis(u)
