# Test https://github.com/wala/ML/issues/1145: a built-in layer creates its variables on its
# first call, so a model whose `call` makes that first call creates them in its trace.
import tensorflow as tf


class DenseModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(4)

    def call(self, x):
        return self.dense(x)


class RecurrentModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.rnn = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(4))

    def call(self, x):
        return self.rnn(x)


class EmbeddingModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(10, 4)

    def call(self, ids):
        return self.embedding(ids)


# Test https://github.com/wala/ML/issues/1168: a nested layer whose only weighted sublayer is an
# `Embedding` creates the embedding's variable in its own trace.
class Embeddings(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.position_embeddings = tf.keras.layers.Embedding(10, 4)

    def call(self, ids):
        return self.position_embeddings(ids)


class NestedEmbeddingModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.emb = Embeddings()

    def call(self, ids):
        return self.emb(ids)


class FlattenModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()

    def call(self, x):
        return self.flatten(x)


dense = DenseModel()
assert not dense.dense.built
tf.function(dense.call)(tf.ones((2, 3)))
assert dense.dense.built
assert [w.shape for w in dense.dense.weights] == [(3, 4), (4,)]

recurrent = RecurrentModel()
tf.function(recurrent.call)(tf.ones((2, 5, 3)))
assert recurrent.rnn.built and len(recurrent.rnn.weights) == 3

embedding = EmbeddingModel()
tf.function(embedding.call)(tf.constant([[1, 2, 3]]))
assert [w.shape for w in embedding.embedding.weights] == [(10, 4)]

nested = NestedEmbeddingModel()
tf.function(nested.emb.call)(tf.constant([[1, 2, 3]]))
assert [w.shape for w in nested.emb.position_embeddings.weights] == [(10, 4)]
nested(tf.constant([[1, 2, 3]]))

flatten = FlattenModel()
tf.function(flatten.call)(tf.ones((2, 3, 4)))
assert flatten.flatten.built and not flatten.flatten.weights
