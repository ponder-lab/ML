# Test `ContainerElements` where the program deletes only a name, `del v`, which removes no list
# element: a list keeps its length.
import tensorflow as tf

x = tf.ones([2, 3])
y = tf.zeros([4], tf.int32)


def sink_list(c):
    assert len(c) == 2


v = [x]
del v
sink_list([x, y])
