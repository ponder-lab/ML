# Test for wala/ML#1134: applying a program-defined Keras layer to a named tuple of Keras symbolic
# tensors runs its `call` on graph tensors, since a named tuple is a nested structure whose fields
# Keras converts, but an attribute of a plain object is not converted.
from collections import namedtuple
from typing import NamedTuple

import tensorflow as tf

SEEN = []
BOXED = []


class Typed(NamedTuple):
    values: tf.Tensor
    masks: tf.Tensor


Plain = namedtuple("Plain", ["values", "masks"])


class Box:

    def __init__(self, t):
        self.t = t


def typed_inside(t):
    SEEN.append(type(t).__name__)
    return t * 2.0


def plain_inside(t):
    SEEN.append(type(t).__name__)
    return t * 2.0


def boxed_inside(t):
    BOXED.append(type(t).__name__)
    return t


class FromTyped(tf.keras.layers.Layer):

    def call(self, inputs):
        return typed_inside(inputs.masks)


class FromPlain(tf.keras.layers.Layer):

    def call(self, inputs):
        return plain_inside(inputs.masks)


class FromBox(tf.keras.layers.Layer):

    def call(self, inputs, box=None):
        # The boxed value is still symbolic here, so computing with it beside `inputs` would raise.
        boxed_inside(box.t)
        return inputs


inp = tf.keras.layers.Input([4])
from_typed = FromTyped()(Typed(inp, inp))
from_plain = FromPlain()(Plain(inp, inp))
from_box = FromBox()(inp, box=Box(inp))
assert SEEN == ["Tensor", "Tensor"], SEEN
# A plain object is not a nested structure, so its attribute stays a symbolic tensor.
assert BOXED == ["KerasTensor"], BOXED
