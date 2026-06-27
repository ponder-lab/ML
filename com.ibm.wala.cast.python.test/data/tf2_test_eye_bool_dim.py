import tensorflow as tf


def consume(z):
    pass


# `tf.eye(True)`: WALA models the Python `bool` as a `Boolean` constant. `getIntValueFromInstanceKey`
# treats it as `int(True) == 1` (degrading gracefully) rather than throwing a `ClassCastException`
# on the `Number` cast, so the identity matrix is `(1, 1)`. wala/ML#590.
consume(tf.eye(True))
