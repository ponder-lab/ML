"""Exercises a class-body PEP-526 annotated assignment with a value (wala/ML#579).

A value-bearing class field ``weight: tf.Tensor = tf.ones([3, 4])`` assigns the class attribute, and reading it back (``y = C.weight``) recovers the ``(3, 4) float32`` type.
This is the value-bearing counterpart to the annotation-only field in ``tf2_test_namedtuple_field.py``.
"""

import tensorflow as tf


def consume(x):
    pass


class C:
    weight: tf.Tensor = tf.ones([3, 4])


y = C.weight
assert y.shape == (3, 4) and y.dtype == tf.float32
consume(y)
