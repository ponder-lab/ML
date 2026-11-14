# Test `ContainerElements` where a function the call graph does not reach, but the program names,
# deletes an element of a parameter, `del l[0]`: the analysis cannot tell which list it is, so no
# list reports a length (wala/ML#1239, wala/ML#1255).
import tensorflow as tf

x = tf.ones([2, 3])
y = tf.zeros([4], tf.int32)


def sink_list(c):
    assert len(c) == 2


def never_called(l):
    del l[0]


handlers = [never_called]

sink_list([x, y])
