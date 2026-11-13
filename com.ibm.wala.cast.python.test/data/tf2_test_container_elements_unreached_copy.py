# Test `ContainerElements` where a function the call graph does not reach resizes a slice copy of a parameter, `a[:]` (wala/ML#1255).
# No class of the program defines `__getitem__`, so the copy is a new list and a list keeps its
# length.
import tensorflow as tf

x = tf.ones([2, 3])
y = tf.zeros([4], tf.int32)


def sink_list(c):
    assert len(c) == 2


def never_called(a):
    copied = a[:]
    copied.append(1)


sink_list([x, y])
