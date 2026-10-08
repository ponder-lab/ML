# Test a stored encoder output written from OUTSIDE the model's own methods (wala/ML#1021): a helper
# taking the model as its SECOND parameter stores a rank-2 tensor under the same attribute the model's
# `call` stores its rank-3 encoder output under. Both stored values are real, so the dense over the
# attribute must read both, `(8, 10, 7)` and `(5, 7)`, while the round trip's dead-arm matrix `(80, 7)`
# must still not appear: a reader that resolves the attribute through the writes it can see must see
# this writer too, or decline, never drop it.
import tensorflow as tf


def get_shape_list(tensor):
    shape = tensor.shape.as_list()
    non_static_indexes = []
    for index, dim in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)
    if not non_static_indexes:
        return shape
    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def reshape_to_matrix(tensor):
    if len(tensor.shape) == 0:
        return tensor
    dim = tensor.shape[-1]
    return tf.reshape(tensor, [-1, dim])


def reshape_from_matrix(output_tensor, orig_shape_list):
    if len(orig_shape_list) == 2:
        return output_tensor
    output_shape = get_shape_list(output_tensor)
    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]
    return tf.reshape(output_tensor, orig_dims + [width])


def consume_logits(t):
    assert t.shape in [(8, 10, 7), (5, 7)]
    return t


class Encoder(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(32)

    def call(self, input_tensor):
        input_shape = get_shape_list(input_tensor)
        matrix = reshape_to_matrix(input_tensor)
        output = self.dense(matrix)
        return reshape_from_matrix(output, input_shape)


class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.head = tf.keras.layers.Dense(7)

    def call(self, x):
        self.sequence_output = self.encoder(x)
        return self.rescore()

    def get_sequence_output(self):
        return self.sequence_output

    def rescore(self):
        logits = self.head(self.get_sequence_output())
        consume_logits(logits)
        return logits


def attach(summary, model):
    model.sequence_output = summary


m = Model()
m(tf.ones((8, 10, 32)))
attach(tf.ones((5, 32)), m)
m.rescore()
