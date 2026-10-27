# Test https://github.com/wala/ML/issues/1166: Keras builds a layer in `Layer.__call__`, before it
# calls `call`, so a subclass's `call` creates no variable, whether or not it runs
# `super().call(...)`, while calling the layer does.
import tensorflow as tf


class BatchNormalization(tf.keras.layers.BatchNormalization):
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


class NormalizedModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.bn = BatchNormalization()

    def call(self, x):
        return self.bn(x)


class TFEmbedding(tf.keras.layers.Embedding):
    def call(self, inputs):
        return tf.gather(self.embeddings, tf.cast(inputs, tf.int32))


class Embeddings(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.position_embeddings = TFEmbedding(10, 4)

    def call(self, ids):
        return self.position_embeddings(ids)


class EmbeddingModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.emb = Embeddings()

    def call(self, ids):
        return self.emb(ids)


class DenseModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(4)

    def call(self, x):
        return self.dense(x)


normalized = NormalizedModel()
assert not normalized.bn.built
tf.function(normalized.call)(tf.random.normal([1, 4, 4, 3]))
assert normalized.bn.built and len(normalized.bn.weights) == 4

# The overriding `call` alone, on a built layer, creates no further variable.
before = len(normalized.bn.weights)
tf.function(normalized.bn.call)(tf.random.normal([1, 4, 4, 3]))
assert len(normalized.bn.weights) == before

embedded = EmbeddingModel()
assert not embedded.emb.position_embeddings.built
tf.function(embedded.emb.call)(tf.constant([[1, 2, 3]]))
assert embedded.emb.position_embeddings.built
assert len(embedded.emb.position_embeddings.weights) == 1
embedded(tf.constant([[1, 2, 3]]))

# A subclass's `call` that skips `super().call(...)` creates no variable on its own either.
before = len(embedded.emb.position_embeddings.weights)
tf.function(embedded.emb.position_embeddings.call)(tf.constant([[1, 2]]))
assert len(embedded.emb.position_embeddings.weights) == before

dense = DenseModel()
tf.function(dense.call)(tf.ones((2, 3)))
assert dense.dense.built and len(dense.dense.weights) == 2
