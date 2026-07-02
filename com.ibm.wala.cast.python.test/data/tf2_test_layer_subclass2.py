# Test for wala/ML#118: a user subclass of `Layer` imported via `from tensorflow.keras.layers
# import Layer` resolves its base class in the class hierarchy instead of falling back to `object`.
from tensorflow.keras.layers import Layer


class MyLayer(Layer):
    def __init__(self):
        super(MyLayer, self).__init__()

    def call(self, inputs):
        return inputs


layer = MyLayer()
assert isinstance(layer, Layer)
