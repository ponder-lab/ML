# A value whose producers share subgraphs: each step adds the previous value to itself, so a read
# that evaluates every operand anew evaluates the first value 2**40 times (wala/ML#1174).
import tensorflow as tf


def consume(x):
    return x


x = tf.ones([2, 3])
x = x + x
x = x + x
x = x + x
x = x + x
x = x + x
x = x + x
x = x + x
x = x + x
x = x + x
x = x + x
x = x + x
x = x + x
x = x + x
x = x + x
x = x + x
x = x + x
x = x + x
x = x + x
x = x + x
x = x + x
x = x + x
x = x + x
x = x + x
x = x + x
x = x + x
x = x + x
x = x + x
x = x + x
x = x + x
x = x + x
x = x + x
x = x + x
x = x + x
x = x + x
x = x + x
x = x + x
x = x + x
x = x + x
x = x + x
x = x + x
consume(x)
assert x.shape == (2, 3)
assert x.dtype == tf.float32
