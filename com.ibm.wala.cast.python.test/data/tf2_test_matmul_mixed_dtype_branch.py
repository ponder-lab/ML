# Test `tf.matmul` reached with operands of different dtypes: `tf.matmul` raises unless both
# operands have one dtype, so an int32 operand against a float32 weight yields no tensor. A layer
# whose `call` dispatches on a `mode` string to an embedding lookup or to a projection reaches the
# projection with its int32 ids as well when the branch is not decided, as a language model's
# shared embedding layer is.
import tensorflow as tf


class Emb(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.w = self.add_weight("w", shape=[10, 8], dtype="float32")

    def call(self, inputs, mode="embedding"):
        if mode == "embedding":
            return self.embed(inputs)
        elif mode == "projection":
            return self.project(inputs)
        else:
            raise ValueError(mode)

    def embed(self, inputs):
        inputs = tf.cast(inputs, tf.int32)
        return tf.nn.embedding_lookup(self.w, inputs)

    def project(self, inputs):
        h = tf.reshape(inputs, [-1, 8])
        return tf.reshape(tf.matmul(h, self.w, transpose_b=True), [2, 3, 10])


def consume_embedded(x):
    assert x.shape == (2, 3, 8) and x.dtype == tf.float32
    return x


def consume_logits(x):
    assert x.shape == (2, 3, 10) and x.dtype == tf.float32
    return x


emb = Emb()
ids = tf.constant([[1, 2, 3], [4, 5, 6]])
hidden = emb(ids)
consume_embedded(hidden)
consume_logits(emb(hidden, mode="projection"))
