import tensorflow as tf
import numpy as np

# Mirrors `evaluate` from `TensorFlow-Examples/.../2_BasicModels/word2vec.py`, a
# real-world word-embedding utility (the cosine similarity between an input
# embedding and every row of the module-level embedding matrix), for tensor-type
# inference coverage.
vocabulary_size = 100
embedding_size = 10

embedding = tf.Variable(tf.random.normal([vocabulary_size, embedding_size]))


def evaluate(x_embed):
    # Compute the cosine similarity between input data embedding and every embedding vector.
    x_embed = tf.cast(x_embed, tf.float32)
    x_embed_norm = x_embed / tf.sqrt(tf.reduce_sum(tf.square(x_embed)))
    embedding_norm = embedding / tf.sqrt(
        tf.reduce_sum(tf.square(embedding), 1, keepdims=True), tf.float32
    )
    cosine_sim_op = tf.matmul(x_embed_norm, embedding_norm, transpose_b=True)
    return cosine_sim_op


x_embed = tf.constant(np.ones((4, embedding_size), dtype=np.float32))
result = evaluate(x_embed)
assert x_embed.shape == (4, embedding_size) and x_embed.dtype == tf.float32
assert result.shape == (4, vocabulary_size) and result.dtype == tf.float32
