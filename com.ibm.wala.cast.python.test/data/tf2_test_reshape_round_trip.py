# The encoder round trip of a transformer layer: flatten the input to a matrix, and reshape the
# layer's output back with the input's leading dims and the output's width, read off shape lists
# (`orig_shape_list[0:-1] + [width]`). The round trip is the identity on the leading dims, and a
# layer loop carrying its output back as the next input keeps the input's shape at every layer.
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


def consume_round_trip(t):
    assert t.shape == (8, 10, 32)
    return t


def consume_layered(t):
    assert t.shape == (8, 10, 32)
    return t


class Layer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(32)

    def call(self, input_tensor):
        input_shape = get_shape_list(input_tensor)
        matrix = reshape_to_matrix(input_tensor)
        output = self.dense(matrix)
        return reshape_from_matrix(output, input_shape)


x = tf.ones((8, 10, 32))
layer = Layer()
consume_round_trip(layer(x))

layers = [Layer() for _ in range(3)]
h = x
for one in layers:
    h = one(h)
consume_layered(h)
