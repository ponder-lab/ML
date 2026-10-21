# Test for wala/ML#1134: applying a program-defined Keras layer to a list or a tuple of Keras
# symbolic tensors runs its `call` on graph tensors, so the elements `call` reads out of its input,
# and the method and functions it passes them to, receive no symbolic tensor.
import tensorflow as tf

SEEN = []


def list_inside(t):
    SEEN.append(type(t).__name__)
    return t * 2.0


def tuple_inside(t):
    SEEN.append(type(t).__name__)
    return t * 2.0


class FromList(tf.keras.layers.Layer):

    def masked(self, inputs, masks):
        SEEN.append(type(masks).__name__)
        return inputs + masks

    def call(self, inputs):
        values, masks = inputs
        return self.masked(list_inside(values), masks)


class FromTuple(tf.keras.layers.Layer):

    def call(self, inputs):
        return tuple_inside(inputs[1])


inp = tf.keras.layers.Input([4])
from_list = FromList()([inp, inp])
from_tuple = FromTuple()((inp, inp))
assert type(from_list).__name__ == "KerasTensor"
assert type(from_tuple).__name__ == "KerasTensor"
assert SEEN == ["Tensor", "Tensor", "Tensor"], SEEN
