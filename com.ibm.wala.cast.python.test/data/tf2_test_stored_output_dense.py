# Test a dense layer over a stored encoder output (wala/ML#1021): the encoder's round trip,
# `reshape_to_matrix`/`reshape_from_matrix`, has guards that are infeasible here, so the value the
# round trip returns is exact, but the dead `return output_tensor` arm leaves the flattened matrix in
# the points-to set of everything downstream: the stored attribute, its getter's result, and the
# dense layer's argument. The dense output must be the exact `(8, 10, 7)` alone, never a second
# `(80, 7)` from the matrix.
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
    assert t.shape == (8, 10, 7)
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
        logits = self.head(self.get_sequence_output())
        consume_logits(logits)
        return tf.reshape(logits, [8, 10, -1])

    def get_sequence_output(self):
        return self.sequence_output


Model()(tf.ones((8, 10, 32)))
