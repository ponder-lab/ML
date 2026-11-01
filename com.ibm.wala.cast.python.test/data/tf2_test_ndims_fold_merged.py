# A rank guard on a value that is a rank-2 tensor on one path and a rank-3 tensor on the other has
# no single rank to decide from, so both of its arms run (wala/ML#1188).
import os

import tensorflow as tf


def consume(x):
    pass


def pick(flag):
    if flag:
        x = tf.ones([2, 2])
    else:
        x = tf.ones([2, 2, 2])
    if x.shape.rank == 3:
        x = tf.ones([4, 2])
    assert x.shape == (2, 2) or x.shape == (4, 2)
    return x


consume(pick(os.environ.get("PICK_CONSTANT") is not None))
