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
