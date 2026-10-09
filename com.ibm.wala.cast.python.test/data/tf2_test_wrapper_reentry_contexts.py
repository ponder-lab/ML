# Test a layer wrapper whose wrapped layer may be another wrapper of its class, as when a field is
# rebound to a wrapper of its old value and the wrappers' constructors merge: each forwarding call
# re-enters the wrapper's `call` on another instance. The call must not take a new context at every
# hop; the hop count bounds the run.
import tensorflow as tf


class LayerWrapper(tf.keras.layers.Layer):
    def __init__(self, layer, hops=2, **kwargs):
        super(LayerWrapper, self).__init__(**kwargs)
        self.layer = layer
        self.hops = hops

    def call(self, inputs, *args, **kwargs):
        if self.hops <= 0:
            return inputs
        self.hops -= 1
        return self.layer(inputs, *args, **kwargs)


class TransformerLayerWrapper(LayerWrapper):
    def __init__(self, layer, **kwargs):
        super(TransformerLayerWrapper, self).__init__(layer, **kwargs)


def consume(x):
    assert x.shape == (2, 5, 8)


def build(flag):
    first = TransformerLayerWrapper(tf.keras.layers.Dense(8))
    second = TransformerLayerWrapper(tf.keras.layers.Dense(8))
    third = TransformerLayerWrapper(tf.keras.layers.Dense(8))
    first.layer = second if flag else third
    second.layer = third if flag else first
    third.layer = first if flag else second
    return first


consume(build(True)(tf.ones((2, 5, 8)), training=False))
