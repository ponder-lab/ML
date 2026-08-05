import numpy as np
import tensorflow as tf


def consume(tensor):
    pass


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d = tf.keras.layers.Dense(10)

    def call(self, x):
        # A Keras layer casts a floating-point input to the layer's compute dtype before the
        # body runs, so this is float32 even though the caller holds float64.
        assert x.dtype == tf.float32
        return self.d(x)


# The image-normalization idiom: NumPy promotes an integral array divided by a Python float
# to float64.
input_data = np.zeros((20, 28), dtype=np.uint8) / 255.0
assert input_data.dtype == np.float64 and input_data.shape == (20, 28)
consume(input_data)

model = MyModel()
result = model(input_data)
assert result.shape == (20, 10)
assert result.dtype == tf.float32
