# Probe driver for wala/ML#618: the vendored `Conv1d` forward output in isolation.
# Analyzed statically, like `A.py` itself.
import tensorflow as tf

from layers.feed_forward import Conv1d


def consume(t):
    pass


c = Conv1d(8, 16)
out = c(tf.ones((2, 3, 8)))
consume(out)
