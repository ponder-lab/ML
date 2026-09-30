# Test for wala/ML#987 and wala/ML#986: a layer that reshapes its result to a shape list built
# from `tf.shape(inputs)` subscripts and a stored attribute, `[tf.shape(inputs)[0],
# tf.shape(inputs)[1]] + [self.filter_size]`, keeps its input's rank, in every instance and in a
# chain of two such layers, and the results of a `tf.split` over the result keep it as well.
import json

import tensorflow as tf


def consume_first(x):
    pass


def consume_second(x):
    pass


def consume_split(x):
    pass


def consume_opaque(x):
    pass


def consume_wide(x):
    pass


def consume_wide_split(x):
    pass


def consume_heads(x):
    pass


class Conv1d(tf.keras.layers.Layer):
    def __init__(self, hidden_size, filter_size):
        super(Conv1d, self).__init__()
        self.hidden_size = hidden_size
        self.filter_size = filter_size

    def build(self, input_shape):
        self.weight = self.add_weight(
            "weight", shape=[self.hidden_size, self.filter_size], initializer="zeros"
        )
        self.bias = self.add_weight(
            "bias", shape=[self.filter_size], initializer="zeros"
        )
        super(Conv1d, self).build(input_shape)

    def call(self, inputs):
        output_shape = [tf.shape(inputs)[0], tf.shape(inputs)[1]] + [self.filter_size]
        inputs = tf.reshape(inputs, [-1, self.hidden_size])
        outputs = tf.matmul(inputs, self.weight) + self.bias
        outputs = tf.reshape(outputs, output_shape)
        return outputs


class Block(tf.keras.layers.Layer):
    def __init__(self):
        super(Block, self).__init__()
        self.dense_layer = Conv1d(8, 32)
        self.output_dense_layer = Conv1d(32, 8)
        self.c_attn = Conv1d(8, 24)
        # A filter size the analysis cannot compute (a configuration value times three, as a
        # transformer's attention projection sizes itself): the rank must survive, with an
        # unresolved last axis.
        self.n_head = 3
        self.wide_size = json.loads("8") * 3
        self.wide = Conv1d(8, self.wide_size)

    def call(self, x):
        first = self.dense_layer(x)
        consume_first(first)
        second = self.output_dense_layer(first)
        consume_second(second)
        query, key, value = tf.split(self.c_attn(x), 3, axis=2)
        consume_split(query)
        wide = self.wide(x)
        consume_wide(wide)
        wq, wk, wv = tf.split(wide, 3, axis=2)
        consume_wide_split(wk)
        # A shape element that is arithmetic over stored attributes, one of them a size the
        # analysis cannot compute (a transformer's `hidden // n_head` head size): the rank must
        # survive, with that axis unresolved.
        heads = tf.reshape(wide, [2, 5, self.n_head, self.wide_size // self.n_head])
        consume_heads(heads)
        return second, key


block = Block()
second, key = block(tf.ones((2, 5, 8)))

# Control: a shape operand opaque to every reader, the value of an unmodeled call with no
# points-to set, still reads as a tensor of unknown rank, as the dataflow's reshape pin read it
# before; the generator's input-shape fallback must not replace that reading.
opaque_shape = json.loads("[80]")
consume_opaque(tf.reshape(second, opaque_shape))
assert second.shape == (2, 5, 8) and second.dtype == tf.float32
assert key.shape == (2, 5, 8)
assert block.wide(tf.ones((2, 5, 8))).shape == (2, 5, 24)
assert tf.reshape(block.wide(tf.ones((2, 5, 8))), [2, 5, 3, 24 // 3]).shape == (
    2,
    5,
    3,
    8,
)
