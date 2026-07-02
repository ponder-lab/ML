# Probe for the collection-dataflow family (wala/ML#618): tensors iterated through `zip` keep
# their types (mirroring gpt-2's `for decoder_layer, past in zip(self.decoder_layers, pasts)`).
import tensorflow as tf


def consume(t):
    pass


xs = [tf.ones((4, 8)), tf.ones((4, 8))]
ys = [tf.zeros((2, 2)), tf.zeros((2, 2))]

for x, y in zip(xs, ys):
    consume(x)
    assert x.shape == (4, 8)
    assert x.dtype == tf.float32
