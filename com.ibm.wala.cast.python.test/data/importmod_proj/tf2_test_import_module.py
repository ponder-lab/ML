# wala/ML#687 MRE: the class is reached through a plain `import B` module object.
import tensorflow.keras as keras

import B


def consume(t):
    pass


input_node = keras.layers.Input(shape=(32, 32, 3))
pad = B.Padding2D()(input_node)
consume(pad)

assert pad.shape.as_list() == [None, 34, 34, 3]
