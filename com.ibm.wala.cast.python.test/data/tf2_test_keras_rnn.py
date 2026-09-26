# Test https://github.com/wala/ML/issues/973: a generic `tf.keras.layers.RNN` over a user cell.
import tensorflow as tf


def consume_step_inputs(x):
    pass


def consume_direct(x):
    pass


def consume_sequence(x):
    pass


def consume_last_state(x):
    pass


def consume_last_output(x):
    pass


class ArgmaxCell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, units, **kwargs):
        super(ArgmaxCell, self).__init__(**kwargs)
        self.units = units

    @property
    def state_size(self):
        return self.units

    @property
    def output_size(self):
        return self.units

    def call(self, inputs, state):
        consume_step_inputs(inputs)
        new_state = inputs + state[0]
        choice = tf.cast(tf.argmax(tf.expand_dims(new_state, 2), 2), dtype=tf.int32)
        return choice, new_state


cell = ArgmaxCell(5)
x = tf.ones((3, 7, 5), dtype=tf.float32)
s0 = tf.zeros((3, 5), dtype=tf.float32)

# Calling the cell directly on one step.
direct, _ = cell(x[:, 0], [s0])
assert direct.shape == (3, 5) and direct.dtype == tf.int32
consume_direct(direct)

# The RNN over the cell: one output per time step, and the last state.
layer = tf.keras.layers.RNN(cell, return_sequences=True, return_state=True)
seq, last = layer(x, s0)
assert seq.shape == (3, 7, 5) and seq.dtype == tf.int32
assert last.shape == (3, 5) and last.dtype == tf.float32
consume_sequence(seq)
consume_last_state(last)

# Without `return_sequences`, the layer's output is the last step's.
last_layer = tf.keras.layers.RNN(cell, return_state=True)
last_out, _ = last_layer(x, s0)
assert last_out.shape == (3, 5) and last_out.dtype == tf.int32
consume_last_output(last_out)
