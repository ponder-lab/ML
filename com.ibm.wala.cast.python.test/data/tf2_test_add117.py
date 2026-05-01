import tensorflow as tf
import random


def add(a, b):
    return a + b


if random.random() < 0.5:
    a = 1
else:
    a = 3

t1 = tf.ones([a, 2])
t2 = tf.ones([2, 2])

assert t1.shape in [(1, 2), (3, 2)]
assert t1.dtype == tf.float32

assert t2.shape == (2, 2)
assert t2.dtype == tf.float32

try:
    c = add(t1, t2)
except tf.errors.InvalidArgumentError:
    pass  # Expected when a=3
