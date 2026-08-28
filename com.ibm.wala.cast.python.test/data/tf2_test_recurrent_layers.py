import tensorflow as tf


def consume_lstm_last(x):
    pass


def consume_lstm_seq(x):
    pass


def consume_gru(x):
    pass


def consume_rnn(x):
    pass


def consume_bidi(x):
    pass


# Applied directly, without a surrounding model: the layer's declared width becomes the output's
# last axis, and `return_sequences` decides whether the temporal axis survives (wala/ML#840).
lstm = tf.keras.layers.LSTM(32)
lstm_out = lstm(tf.ones((8, 10, 4)))
assert lstm_out.shape == (8, 32), lstm_out.shape
assert lstm_out.dtype == tf.float32, lstm_out.dtype
consume_lstm_last(lstm_out)

lstm_seq = tf.keras.layers.LSTM(32, return_sequences=True)
seq_out = lstm_seq(tf.ones((8, 10, 4)))
assert seq_out.shape == (8, 10, 32), seq_out.shape
consume_lstm_seq(seq_out)

gru = tf.keras.layers.GRU(16)
gru_out = gru(tf.ones((8, 10, 4)))
assert gru_out.shape == (8, 16), gru_out.shape
consume_gru(gru_out)

rnn = tf.keras.layers.SimpleRNN(8)
rnn_out = rnn(tf.ones((8, 10, 4)))
assert rnn_out.shape == (8, 8), rnn_out.shape
consume_rnn(rnn_out)

# `Bidirectional` under the default `concat` merge mode doubles the wrapped layer's width.
bidi = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))
bidi_out = bidi(tf.ones((8, 10, 4)))
assert bidi_out.shape == (8, 64), bidi_out.shape
consume_bidi(bidi_out)
