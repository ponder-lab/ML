# A callee whose return has two arms selected by a string mode (wala/ML#958): the embedding arm
# is an elementwise result (no allocation of its own), the projection arm is an int32-typed
# allocation. An add over the callee's result then sees the projection arm's dtype on the
# embedding arm's shape unless the operands' dataflow dtypes decide the add's dtype.
import tensorflow as tf


class Codec(tf.keras.layers.Layer):
    def __init__(self, vocab, width):
        super(Codec, self).__init__()
        self.vocab = vocab
        self.width = width

    def build(self, input_shape):
        self.table = self.add_weight(
            "table", shape=[self.vocab, self.width], dtype="float32"
        )
        super(Codec, self).build(input_shape)

    def call(self, inputs, mode="embedding"):
        if mode == "embedding":
            return self.embed(inputs)
        else:
            return self.project(inputs)

    def embed(self, inputs):
        mask = tf.cast(tf.not_equal(inputs, 0), tf.float32)
        inputs = tf.cast(inputs, tf.int32)
        embeddings = tf.nn.embedding_lookup(self.table, inputs)
        embeddings *= tf.expand_dims(mask, -1)
        return embeddings

    def project(self, inputs):
        flat = tf.reshape(inputs, [-1, self.width])
        logits = tf.matmul(flat, self.table, transpose_b=True)
        return tf.reshape(logits, [2, 3, self.vocab])


def consume(hidden):
    pass


def consume_logits(logits):
    pass


codec = Codec(10, 4)
ids = tf.cast(tf.constant([[1, 2, 3], [4, 5, 6]]), tf.int32)
positions = tf.ones((2, 3, 4), dtype=tf.float32)
hidden = codec(ids) + positions
assert hidden.shape == (2, 3, 4) and hidden.dtype == tf.float32
consume(hidden)
logits = codec(hidden, mode="projection")
assert logits.shape == (2, 3, 10) and logits.dtype == tf.float32
consume_logits(logits)
