# Test https://github.com/wala/ML/issues/1322: a layer whose `call` forwards `*args` (and `**kwargs`)
# to a wrapped Keras layer gives the wrapped layer the argument the starred tuple holds.
import tensorflow as tf


def consume_star(t):
    assert t.shape == (2, 3, 4)


def consume_star_kw(t):
    assert t.shape == (2, 3, 4)


def consume_star_dense(t):
    assert t.shape == (2, 7)


def consume_explicit(t):
    assert t.shape == (2, 3, 4)


class Star(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.rnn = tf.keras.layers.LSTM(4, return_sequences=True)

    def call(self, *args):
        return self.rnn(*args)


class StarKw(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.rnn = tf.keras.layers.LSTM(4, return_sequences=True)

    def call(self, *args, **kwargs):
        return self.rnn(*args, **kwargs)


class StarDense(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(7)

    def call(self, *args):
        return self.dense(*args)


class Explicit(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.rnn = tf.keras.layers.LSTM(4, return_sequences=True)

    def call(self, inputs):
        return self.rnn(inputs)


consume_star(Star()(tf.ones([2, 3, 5])))
consume_star_kw(StarKw()(tf.ones([2, 3, 5]), training=False))
consume_star_dense(StarDense()(tf.ones([2, 5])))
consume_explicit(Explicit()(tf.ones([2, 3, 5])))
