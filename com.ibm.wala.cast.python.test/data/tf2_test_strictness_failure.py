import tensorflow as tf


def test_strictness(t):
    pass


class MyClass:
    pass


obj = MyClass()

try:
    # This should raise an exception because row_lengths must be integers
    t = tf.RaggedTensor.from_row_lengths(values=[1, 2, 3], row_lengths=[obj, 1])
    test_strictness(t)
except Exception:
    # Expected exception caught, script runs to completion
    pass
