import tensorflow as tf


class BiLSTM(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(1000, 64)

    def call(self, inputs):
        # `inputs` receives a token-id tensor at the call site, then feeds an `Embedding`.
        assert inputs.shape == (1, 3)
        assert inputs.dtype == tf.int32
        return self.embedding(inputs)


BiLSTM()(tf.constant([[1, 2, 3]]))
