import tensorflow as tf


def add(a, b):
    return a + b


a = tf.ones([1, 2], tf.float32)
b = tf.ones([2, 2], tf.float32)
assert a.shape == (1, 2), f"Expected shape (1, 2), got {a.shape}"
assert b.shape == (2, 2), f"Expected shape (2, 2), got {b.shape}"

assert a.dtype == tf.float32, f"Expected dtype float32, got {a.dtype}"
assert b.dtype == tf.float32, f"Expected dtype float32, got {b.dtype}"

c = add(a, b)

assert c.shape == (2, 2), f"Expected shape (2, 2), got {c.shape}"
assert c.dtype == tf.float32, f"Expected dtype float32, got {c.dtype}"
