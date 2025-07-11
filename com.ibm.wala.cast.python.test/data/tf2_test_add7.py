import tensorflow as tf


def add(a, b):
    assert a.shape == (1, 2), f"Expected shape (1, 2), got {a.shape}"
    assert b.shape == (2, 2), f"Expected shape (2, 2), got {b.shape}"

    assert a.dtype == tf.float32, f"Expected dtype float32, got {a.dtype}"
    assert b.dtype == tf.float32, f"Expected dtype float32, got {b.dtype}"

    return a + b


c = add(tf.ones([1, 2]), tf.ones([2, 2]))  #  [[2., 2.], [2., 2.]]
