import sys

import numpy as np
import tensorflow as tf


def consume_row_constant(x):
    assert x.shape == (4,)
    assert x.dtype == tf.float32


def consume_row_unknown(x):
    assert x.shape == (4,)
    assert x.dtype == tf.float32


def consume_row_enumerate(x):
    assert x.shape == (4,)
    assert x.dtype == tf.float32


def consume_row_parameter(x):
    assert x.shape == (4,)
    assert x.dtype == tf.float32


def consume_row_sliced(x):
    assert x.shape == (2,)
    assert x.dtype == tf.float32


def consume_element_unknown(x):
    assert x.shape == (4,)
    assert x.dtype == tf.float32


def consume_array_row_unknown(x):
    assert x.shape == (4,)
    assert x.dtype == np.float64


def consume_column_unknown(x):
    assert x.shape == (3,)
    assert x.dtype == tf.float32


def read_row(states, index):
    consume_row_parameter(states[index, :])


states = tf.ones((3, 4))

zero = 0
consume_row_constant(states[zero, :])

unknown = len(sys.argv) - 1
consume_row_unknown(states[unknown, :])

for position, element in enumerate([states, states]):
    consume_row_enumerate(states[position, :])
    read_row(states, position)

consume_row_sliced(states[unknown, 1:3])
consume_element_unknown(states[unknown])
consume_array_row_unknown(np.ones((3, 4))[unknown, :])
consume_column_unknown(states[:, unknown])
