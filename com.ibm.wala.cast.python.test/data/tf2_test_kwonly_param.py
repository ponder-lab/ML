import tensorflow as tf


def consume(t):
    pass


def f(x, *, y):
    # `y` is a keyword-only parameter (after the bare `*`). It must be modeled as a
    # formal parameter so the call-site keyword argument binds to it (wala/ML#596).
    consume(y)


f(tf.constant(1), y=tf.ones([2, 3]))
