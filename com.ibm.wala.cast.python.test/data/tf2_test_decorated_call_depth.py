import tensorflow as tf


def g(b):
    pass


@tf.function
def f():
    a = tf.constant(5)
    # `a` is `g`'s argument here (calling context for the FUT `g`).
    assert a.shape == ()
    assert a.dtype == tf.int32
    g(a)


f()
