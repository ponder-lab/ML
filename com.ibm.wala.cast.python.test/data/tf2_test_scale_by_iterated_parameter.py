# A tensor scaled by the elements of a parameter that one caller passes a `range` and another a
# value of unknown kind: not every caller passes a `range`, so the element is not shown to be a
# Python number (wala/ML#1039).
import json

import tensorflow as tf


def consume_each(t):
    pass


def scale_each(x, steps):
    for step in steps:
        product = x * step
        assert product.shape == (3,) and product.dtype == tf.float32
        consume_each(product)


scale_each(tf.ones([3]), range(2))
scale_each(tf.ones([3]), json.loads("[0, 1]"))
