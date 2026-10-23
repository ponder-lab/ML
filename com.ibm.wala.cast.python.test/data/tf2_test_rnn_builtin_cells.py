# Test a generic `tf.keras.layers.RNN` over a built-in cell, and a direct call on one: the cell's
# step output is `(batch, units)` in the layer's float32 compute dtype, and the layer reinserts the
# time axis.
import tensorflow as tf
import tensorflow_addons as tfa


def consume_simple(x):
    pass


def consume_gru(x):
    pass


def consume_lstm(x):
    pass


def consume_peephole(x):
    pass


def consume_cell_output(x):
    pass


def consume_cell_state(x):
    pass


def consume_lstm_cell_carry(x):
    pass


def consume_subclass_sequence(x):
    pass


def consume_subclass_direct(x):
    pass


class MyCell(tf.keras.layers.SimpleRNNCell):
    def call(self, inputs, states):
        out = tf.cast(tf.reduce_sum(inputs, axis=-1, keepdims=True) > 0, tf.int32)
        return out, states


inputs = tf.zeros((2, 3, 5))
state = tf.zeros((2, 4))

simple = tf.keras.layers.RNN(
    tf.keras.layers.SimpleRNNCell(4), return_state=True, return_sequences=True
)
simple_output = simple(inputs=inputs, initial_state=state)[0]
assert simple_output.shape == (2, 3, 4) and simple_output.dtype == tf.float32
consume_simple(simple_output)

gru = tf.keras.layers.RNN(
    tf.keras.layers.GRUCell(4), return_state=True, return_sequences=True
)
gru_output = gru(inputs=inputs, initial_state=state)[0]
assert gru_output.shape == (2, 3, 4) and gru_output.dtype == tf.float32
consume_gru(gru_output)

lstm = tf.keras.layers.RNN(
    tf.keras.layers.LSTMCell(4), return_state=True, return_sequences=True
)
lstm_output = lstm(inputs=inputs, initial_state=(state, state))[0]
assert lstm_output.shape == (2, 3, 4) and lstm_output.dtype == tf.float32
consume_lstm(lstm_output)

peephole = tf.keras.layers.RNN(
    tfa.rnn.PeepholeLSTMCell(4), return_state=True, return_sequences=True
)
peephole_output = peephole(inputs=inputs, initial_state=(state, state))[0]
assert peephole_output.shape == (2, 3, 4) and peephole_output.dtype == tf.float32
consume_peephole(peephole_output)

# Calling a cell directly on one step: `(output, new_states)`.
cell = tf.keras.layers.SimpleRNNCell(4)
cell_output, cell_states = cell(inputs[:, 0, :], [state])
assert cell_output.shape == (2, 4) and cell_output.dtype == tf.float32
assert cell_states[0].shape == (2, 4) and cell_states[0].dtype == tf.float32
consume_cell_output(cell_output)
consume_cell_state(cell_states[0])

lstm_cell = tf.keras.layers.LSTMCell(4)
_, lstm_cell_states = lstm_cell(inputs[:, 0, :], [state, state])
assert lstm_cell_states[1].shape == (2, 4) and lstm_cell_states[1].dtype == tf.float32
consume_lstm_cell_carry(lstm_cell_states[1])

# A subclass of a built-in cell that overrides `call`: its own `call` decides the step.
subclass_output = tf.keras.layers.RNN(
    MyCell(4), return_state=True, return_sequences=True
)(inputs=inputs, initial_state=state)[0]
assert subclass_output.shape == (2, 3, 1) and subclass_output.dtype == tf.int32
consume_subclass_sequence(subclass_output)

subclass_direct = MyCell(4)(inputs[:, 0, :], [state])[0]
assert subclass_direct.shape == (2, 1) and subclass_direct.dtype == tf.int32
consume_subclass_direct(subclass_direct)
