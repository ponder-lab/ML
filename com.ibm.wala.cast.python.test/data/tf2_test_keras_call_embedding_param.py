import tensorflow as tf


class BiLSTM(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(1000, 64)

    def call(self, inputs):
        return self.embedding(inputs)


# `arg` is the token-id tensor passed to `BiLSTM.call` via `__call__`, then fed to an `Embedding`.
arg = tf.constant([[1, 2, 3]])
assert arg.shape == (1, 3)
assert arg.dtype == tf.int32
BiLSTM()(arg)
