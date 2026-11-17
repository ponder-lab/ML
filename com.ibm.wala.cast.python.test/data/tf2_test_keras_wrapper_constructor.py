# Test https://github.com/wala/ML/issues/1288: the constructor of `tf.keras.layers.Wrapper` stores
# the layer a subclass passes it as `self.layer`, so the subclass's `call` reaches the wrapped layer,
# and a store into one of its attributes writes that layer. `Held` stores the layer itself, as a
# control.
import tensorflow as tf


def consume_wide(p):
    assert p.shape == (2, 8)
    assert p.dtype == tf.float32


def consume_narrow(p):
    assert p.shape == (2, 3)
    assert p.dtype == tf.float32


def consume_held(p):
    assert p.shape == (2, 5)
    assert p.dtype == tf.float32


class Wrapped(tf.keras.layers.Wrapper):
    # The wrapped layer is stored by the Keras base class.
    def __init__(self, layer):
        super().__init__(layer)

    def build(self, input_shape=None):
        if not self.layer.built:
            self.layer.build(input_shape)
            self.v = self.layer.kernel
        super().build()

    def call(self, inputs):
        self.layer.kernel = tf.nn.l2_normalize(self.v, axis=0)
        return self.layer(inputs)


class Held(tf.keras.layers.Layer):
    # The wrapped layer is stored by this class.
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def build(self, input_shape=None):
        if not self.layer.built:
            self.layer.build(input_shape)
            self.v = self.layer.kernel
        super().build(input_shape)

    def call(self, inputs):
        self.layer.kernel = tf.nn.l2_normalize(self.v, axis=0)
        return self.layer(inputs)


# Two instances around different layers, so each `self.layer` must be its own instance's.
consume_wide(Wrapped(tf.keras.layers.Dense(8))(tf.ones([2, 4])))
consume_narrow(Wrapped(tf.keras.layers.Dense(3))(tf.ones([2, 4])))
consume_held(Held(tf.keras.layers.Dense(5))(tf.ones([2, 4])))
