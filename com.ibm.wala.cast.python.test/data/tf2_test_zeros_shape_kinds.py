# An allocator whose shape argument is not provably an `int`: one of unknown kind, and one that may
# be an `int` or a tensor's static shape. Either gives an unknown shape (wala/ML#1159).
import json

import numpy as np
import tensorflow as tf


def consume_unknown(m):
    pass


def consume_int_or_shape(m):
    pass


unknown = np.zeros(json.loads("3"))
assert unknown.shape == (3,) and unknown.dtype == np.float64
consume_unknown(unknown)

xs = [1, 2, 3]
t = tf.ones([2, 3])
for flag in (True, False):
    either = len(xs) if flag else t.shape
    zeros = np.zeros(either)
    assert zeros.shape == ((3,) if flag else (2, 3)) and zeros.dtype == np.float64
    consume_int_or_shape(zeros)
