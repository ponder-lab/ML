# wala/ML#687 control: the same class reached through `from B import Padding2D`.
import tensorflow.keras as keras

from B import Padding2D


def consume(t):
    pass


input_node = keras.layers.Input(shape=(32, 32, 3))
pad = Padding2D()(input_node)
consume(pad)

assert pad.shape.as_list() == [None, 34, 34, 3]
