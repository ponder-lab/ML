import tensorflow as tf


def consume(t):
    pass


def get_shape_list(tensor):
    return tensor.shape.as_list()


# NLPGNN's WDEmbedding in miniature (wala/ML#711): the output reshape's target
# slices the unknown-arity input shape, so the walk fails and the output shape
# is soundly unknown, but the embedding table's dtype must survive through
# both the `gather` arm and the `one_hot`/`matmul` arm.
class WDEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_size, use_one_hot_embedding, **kwargs):
        super(WDEmbedding, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.use_one_hot_embedding = use_one_hot_embedding

    def build(self, input_shape):
        self.embedding_table = self.add_weight(
            name="embeddings",
            shape=[self.vocab_size, self.embedding_size],
            trainable=True,
        )
        self.built = True

    def call(self, input_ids):
        if input_ids.shape.ndims == 2:
            input_ids = tf.expand_dims(input_ids, axis=[-1])
        flat_input_ids = tf.reshape(input_ids, [-1])
        if self.use_one_hot_embedding:
            one_hot_input_ids = tf.keras.backend.one_hot(
                flat_input_ids, self.vocab_size
            )
            output = tf.linalg.matmul(one_hot_input_ids, self.embedding_table)
        else:
            output = tf.gather(self.embedding_table, flat_input_ids)
        input_shape = get_shape_list(input_ids)
        output = tf.reshape(
            output, input_shape[0:-1] + [input_shape[-1] * self.embedding_size]
        )
        return output


layer = WDEmbedding(10, 8, False)
ids = tf.constant([[1, 2], [3, 4]])
out = layer(ids)

assert out.shape == (2, 2, 8)
assert out.dtype == tf.float32

consume(out)
