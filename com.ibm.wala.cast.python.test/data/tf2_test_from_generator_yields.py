"""Witness for wala/ML#903: a from_generator dataset with output_types and NO output_shapes, whose
generator yields (np.array(seq), np.array(label)) with seq an np.pad result and label a scalar; mirrors
the dynamic_rnn.py pipeline exactly (repeat, shuffle, batch(batch_size), the LSTM model with a reshape
[-1, seq_max_len, 1], the three training functions, the loop). Reading the yields types the label
component as a scalar, so the batched labels are (64,); the seq component stays unknown because np.pad is
not modelled, so the batched sequences and the predictions keep their unknown extents.
"""

import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers

num_classes = 2
seq_max_len = 20
seq_min_len = 5
masking_val = -1
max_value = 10000
num_units = 4
batch_size = 64
training_steps = 1
learning_rate = 0.001


def toy_sequence_data():
    while True:
        seq_len = random.randint(seq_min_len, seq_max_len)
        rand_start = random.randint(0, max_value - seq_len)
        if random.random() < 0.5:
            seq = np.arange(start=rand_start, stop=rand_start + seq_len)
            seq = seq / max_value
            seq = np.pad(
                seq,
                mode="constant",
                pad_width=(0, seq_max_len - seq_len),
                constant_values=masking_val,
            )
            label = 0
        else:
            seq = np.random.randint(max_value, size=seq_len)
            seq = seq / max_value
            seq = np.pad(
                seq,
                mode="constant",
                pad_width=(0, seq_max_len - seq_len),
                constant_values=masking_val,
            )
            label = 1
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


optimizer = tf.optimizers.Adam(learning_rate)


def run_optimization(x, y):
    with tf.GradientTape() as g:
        pred = lstm_net(x, is_training=True)
        loss = cross_entropy_loss(pred, y)
    trainable_variables = lstm_net.trainable_variables
    gradients = g.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))


for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    assert batch_x.shape == (64, 20)
    assert batch_y.shape == (64,)
    run_optimization(batch_x, batch_y)
    pred = lstm_net(batch_x, is_training=True)
    assert pred.shape == (64, 2)
    loss = cross_entropy_loss(pred, batch_y)
    acc = accuracy(pred, batch_y)
