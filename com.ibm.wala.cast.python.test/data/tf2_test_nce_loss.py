import tensorflow as tf
import numpy as np

# Mirrors `nce_loss` from `TensorFlow-Examples/.../2_BasicModels/word2vec.py`, a
# real-world word-embedding utility (the averaged noise-contrastive-estimation
# loss over module-level embedding/weight/bias variables), for tensor-type
# inference coverage.
vocabulary_size = 100
embedding_size = 10
num_sampled = 8

embedding = tf.Variable(tf.random.normal([vocabulary_size, embedding_size]))
nce_weights = tf.Variable(tf.random.normal([vocabulary_size, embedding_size]))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))


def nce_loss(x_embed, y):
    y = tf.cast(y, tf.int64)
    loss = tf.reduce_mean(
        tf.nn.nce_loss(
            weights=nce_weights,
            biases=nce_biases,
            labels=y,
            inputs=x_embed,
            num_sampled=num_sampled,
            num_classes=vocabulary_size,
        )
    )
    return loss


x_embed = tf.constant(np.ones((4, embedding_size), dtype=np.float32))
y = tf.constant(np.ones((4, 1), dtype=np.int32))
result = nce_loss(x_embed, y)
assert x_embed.shape == (4, embedding_size) and x_embed.dtype == tf.float32
assert y.shape == (4, 1) and y.dtype == tf.int32
assert result.shape == () and result.dtype == tf.float32
