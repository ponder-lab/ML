# Test https://github.com/wala/ML/issues/1023: a base constructor reached through an explicit
# `super(...).__init__(...)` from a subclass constructor writes to the one instance under
# construction, so two wrappers built around different layers each forward to their own layer.
import tensorflow as tf


def consume_a(p):
    assert p.shape == (2, 8)
    return p


def consume_b(p):
    assert p.shape == (2, 3)
    return p


class LayerWrapper(tf.keras.layers.Layer):
    def __init__(self, layer, **kwargs):
        super(LayerWrapper, self).__init__(**kwargs)
        self.layer = layer

    def call(self, inputs):
        return self.layer(inputs)


class TransformerLayerWrapper(LayerWrapper):
    def __init__(self, layer, output_dropout, **kwargs):
        super(TransformerLayerWrapper, self).__init__(layer, **kwargs)
        self.output_dropout = output_dropout


class WrapperTest:
    def test_wrappers(self):
        a = TransformerLayerWrapper(tf.keras.layers.Dense(8), 0.1)
        b = TransformerLayerWrapper(tf.keras.layers.Dense(3), 0.1)
        x = tf.ones((2, 5))
        consume_a(a(x))
        consume_b(b(x))


WrapperTest().test_wrappers()
