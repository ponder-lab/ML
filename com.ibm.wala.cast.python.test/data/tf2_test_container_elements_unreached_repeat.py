# Test `ContainerElements` where a function the call graph does not reach resizes a repetition of a list it builds, `[1] * n` (wala/ML#1255).
# The repetition is a new list, so a list keeps its length.
import tensorflow as tf

x = tf.ones([2, 3])
y = tf.zeros([4], tf.int32)


def sink_list(c):
    assert len(c) == 2


def never_called(n):
    repeated = [1] * n
    repeated.append(1)


sink_list([x, y])
