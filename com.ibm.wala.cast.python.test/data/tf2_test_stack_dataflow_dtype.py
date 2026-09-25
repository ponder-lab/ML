import tensorflow as tf


def consume(t):
    assert t.shape == (2, 2, 4) and t.dtype == tf.float32


class Block(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.w = self.add_weight("w", shape=[4, 4], dtype=tf.float32)
        super(Block, self).build(input_shape)

    def call(self, x):
        return tf.matmul(x, self.w)


class Pair(tf.keras.Model):
    def __init__(self):
        super(Pair, self).__init__()
        self.key = Block()
        self.value = Block()

    def call(self, x):
        k = self.key(x)
        v = self.value(x)
        present = tf.stack([k, v], axis=1)
        consume(present)
        return present


model = Pair()
out = model(tf.ones((2, 4)))
assert out.shape == (2, 2, 4) and out.dtype == tf.float32
