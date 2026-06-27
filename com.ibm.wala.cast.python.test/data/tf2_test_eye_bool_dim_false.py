import tensorflow as tf


def consume(z):
    pass


# `tf.eye(False)`: the Boolean degrades to `int(False) == 0` (the companion of `tf.eye(True)`),
# yielding a `(0, 0)` identity rather than throwing a `ClassCastException` on the `Number` cast.
# wala/ML#590.
consume(tf.eye(False))
