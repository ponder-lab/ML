import tensorflow as tf


def consume(z):
    pass


# `tf.fill(x.shape, v)` takes its shape from `x`. Regression guard for wala/ML#610:
# the `.shape` dims argument is recovered to `(2, 2)` rather than dropping to ⊤.
x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
consume(tf.fill(x.shape, 5.0))
