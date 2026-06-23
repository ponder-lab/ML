"""Exercises tensor-type propagation through a ``NamedTuple`` field declared with a *dotted* base (``typing.NamedTuple``) rather than a bare ``NamedTuple`` (https://github.com/wala/ML/issues/571).

Identical to ``tf2_test_namedtuple_field.py`` except the base class is written as the attribute chain ``typing.NamedTuple``. This pins the dotted-base path of ``PythonConstructorTargetSelector.isPositionalFieldClass``: only once ``getMissingTypeNames()`` carries the full ``typing.NamedTuple`` (not just the root ``typing``) does the positional-field synthesis fire, so ``b = w.tensor`` keeps its ``(4, 8) float32`` type.
"""

import typing

import numpy as np
import tensorflow as tf


class Wrapper(typing.NamedTuple):
    tensor: tf.Tensor
    rest: typing.List


def consume(x):
    pass


a = tf.constant(np.ones((4, 8), dtype=np.float32))
assert a.shape == (4, 8) and a.dtype == tf.float32
w = Wrapper(a, [])
b = w.tensor
assert b.shape == (4, 8) and b.dtype == tf.float32
consume(b)
