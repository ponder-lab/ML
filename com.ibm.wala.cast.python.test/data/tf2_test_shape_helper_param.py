import tensorflow as tf


def f(a):
    pass


def get_shape_list(tensor, expected_rank=None, name=None):
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
    tensor_2d = tf.reshape(tensor, [-1, dim])
    return tensor_2d


def reshape_from_matrix(output_tensor, orig_shape_list):
    if len(orig_shape_list) == 2:
        return output_tensor
    output_shape = get_shape_list(output_tensor)
    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]
    return tf.reshape(output_tensor, orig_dims + [width])


# The BERT matrix-reshape round trip vendored from NLPGNN's `nlpgnn/tools.py`
# (wala/ML#706): `reshape_from_matrix`'s reshape target is `orig_dims +
# [width]`, where `orig_dims` slices the `orig_shape_list` PARAMETER — the
# def-use walk roots at a parameter and must map it back to the corresponding
# argument at each caller's invoke (here the `get_shape_list(t)` chain) and
# continue in the caller's frame.
t = tf.ones((4, 5, 6))
input_shape = get_shape_list(t)
m = reshape_to_matrix(t)
assert m.shape == (20, 6)
r = reshape_from_matrix(m, input_shape)
assert isinstance(r, tf.Tensor)
assert r.shape == (4, 5, 6)
assert r.dtype == tf.float32
f(r)
