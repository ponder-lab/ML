import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers

# The variable-length sequence classifier in miniature, mirroring its subject shape: a Python
# generator feeding `from_generator` WITHOUT `output_shapes`, so the dataset element's static shape
# is unknown even at run time (TensorFlow reports `<unknown>` rank there), and the model's own
# `reshape` re-establishes rank internally. The prediction that flows back out therefore has a
# statically provable shape, `(None, 2)` after the final Dense, while the raw batch does not.

num_classes = 2
seq_max_len = 20
seq_min_len = 5
masking_val = -1
max_value = 10000
batch_size = 64
num_units = 32


def toy_sequence_data():
    while True:
        seq_len = random.randint(seq_min_len, seq_max_len)
        rand_start = random.randint(0, max_value - seq_len)
        seq = np.arange(start=rand_start, stop=rand_start + seq_len)
        seq = seq / max_value
        seq = np.pad(
            seq,
            mode="constant",
            pad_width=(0, seq_max_len - seq_len),
            constant_values=masking_val,
        )
        label = 0
        yield np.array(seq, dtype=np.float32), np.array(label, dtype=np.float32)


train_data = tf.data.Dataset.from_generator(
    toy_sequence_data, output_types=(tf.float32, tf.float32)
)
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)


class LSTM(Model):
    def __init__(self):
        super(LSTM, self).__init__()
        self.masking = layers.Masking(mask_value=masking_val)
        self.lstm = layers.LSTM(units=num_units)
        self.out = layers.Dense(num_classes)

    def call(self, x, is_training=False):
        x = tf.reshape(x, shape=[-1, seq_max_len, 1])
        x = self.masking(x)
        x = self.lstm(x)
        x = self.out(x)
        if not is_training:
            x = tf.nn.softmax(x)
        return x


lstm_net = LSTM()


def cross_entropy_loss(x, y):
    y = tf.cast(y, tf.int64)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    return tf.reduce_mean(loss)


def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


for step, (batch_x, batch_y) in enumerate(train_data.take(1), 1):
    pred = lstm_net(batch_x, is_training=True)
    # The model's internal reshape re-establishes rank, so the prediction's static shape is
    # (None, 2) even though batch_x's is unknown; at run time the batch is full.
    assert pred.shape == (batch_size, num_classes), pred.shape
    assert pred.dtype == tf.float32, pred.dtype
    loss = cross_entropy_loss(pred, batch_y)
    acc = accuracy(pred, batch_y)
    assert loss.shape == (), loss.shape
    assert acc.shape == (), acc.shape
