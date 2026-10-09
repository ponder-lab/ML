# Test SentencePiece's id encoding feeding a text generator's model: `encode_as_ids` (and its
# `EncodeAsIds` and `encode` spellings) returns a list of Python ints, so the prompt
# `tf.expand_dims([bos] + sp.encode_as_ids(text), 0)` is an int32 `(1, n)` tensor, and the model's
# input is that prompt or the int32 `tf.random.categorical` draw fed back. A concatenation of two
# int literals' lists converts to int32 as well.
import os
import tempfile

import sentencepiece as spm
import tensorflow as tf


def consume(x):
    assert x.dtype == tf.int32 and x.shape[0] == 1
    return x


def consume_encode(x):
    assert x.dtype == tf.int32 and len(x.shape) == 1
    return x


def consume_encode_as_ids_capitalized(x):
    assert x.dtype == tf.int32 and len(x.shape) == 1
    return x


def consume_literal_concat(x):
    assert x.dtype == tf.int32 and x.shape == (1, 3)
    return x


def consume_other_encode(x):
    assert x.dtype == tf.float32 and x.shape == (2,)
    return x


class OtherEncoder(object):
    """An unrelated encoder whose `encode` returns floats: only SentencePiece's yields ids."""

    def encode(self, text):
        return [0.5, 1.5]


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.embedding = tf.keras.layers.Embedding(64, 4)
        self.dense = tf.keras.layers.Dense(10)

    def call(self, x, training=True, past=None):
        consume(x)
        x = tf.cast(x, tf.int32)
        return self.dense(self.embedding(x)), past


class Sampler(object):
    def __init__(self, model_path):
        self.model = Model()
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def sample_sequence(self, context=None, seq_len=3):
        context = tf.expand_dims(([3] + self.sp.encode_as_ids(context)), 0)
        prev = context
        past = None
        for i in range(seq_len):
            logits, past = self.model(prev, training=False, past=past)
            logits = logits[:, -1, :] / tf.cast(1, tf.float32)
            samples = tf.random.categorical(logits, num_samples=1, dtype=tf.int32)
            prev = samples

    def encodings(self, text):
        consume_encode(tf.constant(self.sp.encode(text)))
        consume_encode_as_ids_capitalized(tf.constant(self.sp.EncodeAsIds(text)))


directory = tempfile.mkdtemp()
corpus = os.path.join(directory, "corpus.txt")
with open(corpus, "w") as f:
    f.write("hello world\nthe quick brown fox jumps over the lazy dog\n" * 50)
spm.SentencePieceTrainer.train(
    input=corpus, model_prefix=os.path.join(directory, "m"), vocab_size=30
)
sampler = Sampler(os.path.join(directory, "m.model"))
sampler.sample_sequence("hello world")
sampler.encodings("the quick brown fox")
consume_literal_concat(tf.expand_dims(([3] + [5, 7]), 0))
consume_other_encode(tf.constant(OtherEncoder().encode("hello")))
