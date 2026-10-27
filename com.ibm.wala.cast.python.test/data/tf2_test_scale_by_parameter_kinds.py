# A tensor scaled by a parameter that one caller passes a number computed from a `range` element
# and another a value of unknown kind: the parameter is not shown to be a Python number, since not
# every caller passes one (wala/ML#1039).
import json

import tensorflow as tf


def consume_counted(t):
    pass


def consume_loaded(t):
    pass


def scale(x, n):
    return x * n


for i in range(2):
    counted = scale(tf.ones([3]), i * 2)
    assert counted.shape == (3,) and counted.dtype == tf.float32
    consume_counted(counted)

loaded = scale(tf.ones([3]), json.loads("2"))
assert loaded.shape == (3,) and loaded.dtype == tf.float32
consume_loaded(loaded)
