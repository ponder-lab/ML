# Probe for the collection-dataflow family (wala/ML#618): a user layer whose `call` returns a
# TUPLE (mirroring gpt-2's `return logits, presents`), unpacked at the nested call site.
import tensorflow as tf


def consume(t):
    pass


class Inner(tf.keras.layers.Layer):
    def __init__(self):
        super(Inner, self).__init__()

    def call(self, inputs):
        logits = tf.linalg.matmul(inputs, inputs)
        present = tf.zeros((2, 2))
        return logits, present


class Outer(tf.keras.layers.Layer):
    def __init__(self):
        super(Outer, self).__init__()
        self.inner = Inner()

    def call(self, inputs):
        x, present = self.inner(inputs)
        consume(x)
        return x


outer = Outer()
out = outer(tf.ones((4, 4)))
assert out.shape == (4, 4)
assert out.dtype == tf.float32
