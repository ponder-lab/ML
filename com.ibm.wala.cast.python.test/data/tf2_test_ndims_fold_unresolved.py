import tensorflow as tf


def consume(x):
    pass


def guarded(x):
    if x.shape.ndims == 2:
        x = tf.expand_dims(x, axis=-1)
    return x


def run():
    # A list of two different-rank tensors gives `t` the union of both ranks, so
    # `x.shape.ndims` has no single statically-known value. The fold must decline
    # and keep both arms rather than pick a rank.
    tensors = [tf.ones([8, 10]), tf.ones([8, 10, 1])]
    for t in tensors:
        y = guarded(t)
        assert y.shape == (8, 10, 1)
        consume(y)


run()
