# Test https://github.com/wala/ML/issues/1252: an int that may be a constant or an int of unknown
# value is no single constant, so a size computed from it is not folded.
import sys

import tensorflow as tf


def consume_mixed(x):
    assert x.shape == (3,)


def consume_mixed_len(x):
    assert x.shape == (4,)


v = 0 if len(sys.argv) > 99 else int(len(sys.argv))
consume_mixed(tf.ones([v + 2]))

w = 0 if len(sys.argv) > 99 else len([1.0, 2.0])
consume_mixed_len(tf.ones([w + 2]))


def consume_counter(x):
    assert x.shape in {(2,), (3,), (4,)}


i = 0
while i < 3:
    consume_counter(tf.ones([i + 2]))
    i += 1
