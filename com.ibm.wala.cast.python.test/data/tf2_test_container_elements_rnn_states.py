# The state list a Keras `RNN` hands its cell, per form of the initial state (wala/ML#1265).
import sys

import tensorflow as tf


def sink_tensor(state):
    assert isinstance(state, (list, tuple)) and len(state) == 1
    assert state[0].shape == (2, 3) and state[0].dtype == tf.float32


def sink_tuple(state):
    assert isinstance(state, (list, tuple)) and len(state) == 2
    assert state[0].shape == (2, 3) and state[0].dtype == tf.float32
    assert state[1].shape == (2, 4) and state[1].dtype == tf.float32


def sink_list(state):
    assert isinstance(state, (list, tuple)) and len(state) == 2
    assert state[0].shape == (2, 3) and state[0].dtype == tf.float32
    assert state[1].shape == (2, 4) and state[1].dtype == tf.float32


def sink_mixed(state):
    assert isinstance(state, (list, tuple)) and len(state) in (1, 2)
    assert state[0].shape == (2, 3) and state[0].dtype == tf.float32


def sink_unpassed(state):
    assert isinstance(state, (list, tuple)) and len(state) == 1
    assert state[0].shape == (2, 3) and state[0].dtype == tf.float32


class TensorCell(tf.keras.layers.AbstractRNNCell):

    @property
    def state_size(self):
        return 3

    @property
    def output_size(self):
        return 3

    def call(self, inputs, state):
        sink_tensor(state)
        s = state[0] + inputs
        return s, [s]


class TupleCell(tf.keras.layers.AbstractRNNCell):

    @property
    def state_size(self):
        return [3, 4]

    @property
    def output_size(self):
        return 3

    def call(self, inputs, state):
        sink_tuple(state)
        s = state[0] + inputs
        return s, [s, state[1]]


class ListCell(tf.keras.layers.AbstractRNNCell):

    @property
    def state_size(self):
        return [3, 4]

    @property
    def output_size(self):
        return 3

    def call(self, inputs, state):
        sink_list(state)
        s = state[0] + inputs
        return s, [s, state[1]]


class MixedCell(tf.keras.layers.AbstractRNNCell):

    @property
    def state_size(self):
        return 3

    @property
    def output_size(self):
        return 3

    def call(self, inputs, state):
        sink_mixed(state)
        s = state[0] + inputs
        return s, [s]


class UnpassedCell(tf.keras.layers.AbstractRNNCell):

    @property
    def state_size(self):
        return 3

    @property
    def output_size(self):
        return 3

    def call(self, inputs, state):
        sink_unpassed(state)
        s = state[0] + inputs
        return s, [s]


x = tf.ones([2, 5, 3])
h = tf.ones([2, 3])
c = tf.ones([2, 4])

tf.keras.layers.RNN(TensorCell(), return_sequences=True, return_state=True)(x, h)
tf.keras.layers.RNN(TupleCell(), return_sequences=True, return_state=True)(
    x, initial_state=(h, c)
)
tf.keras.layers.RNN(ListCell(), return_sequences=True, return_state=True)(
    x, initial_state=[h, c]
)
# One call site passing a tensor or a tuple: only the tensor is passed at run time.
mixed = h if len(sys.argv) < 100 else (h, c)
tf.keras.layers.RNN(MixedCell(), return_sequences=True, return_state=True)(x, mixed)
tf.keras.layers.RNN(UnpassedCell(), return_sequences=True, return_state=True)(x)
