# Test `ContainerElements` where the program deletes an element of a list an unmodeled call returns,
# `del random.choice([xs])[0]`: the analysis cannot tell which list it is, so no list reports a
# length (wala/ML#1239).
import random

import tensorflow as tf

x = tf.ones([2, 3])
y = tf.zeros([4], tf.int32)


def sink_list(c):
    assert len(c) == 1


xs = [x, y]
del random.choice([xs])[0]
sink_list(xs)
