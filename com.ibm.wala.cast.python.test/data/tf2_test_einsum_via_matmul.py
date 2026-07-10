import numpy as np
import tensorflow as tf


def consume(t):
    pass


def get_shape_list(tensor):
    return tensor.shape.as_list()


def einsum_via_matmul(input_tensor, w, num_inner_dims):
    input_shape = get_shape_list(input_tensor)
    w_shape = get_shape_list(w)
    batch_dims = input_shape[:-num_inner_dims]
    inner_dims = input_shape[-num_inner_dims:]
    outer_dims = w_shape[num_inner_dims:]
    inner_dim = np.prod(inner_dims)
    outer_dim = np.prod(outer_dims)
    if num_inner_dims > 1:
        input_tensor = tf.reshape(input_tensor, batch_dims + [inner_dim])
    if len(w_shape) > 2:
        w = tf.reshape(w, [inner_dim, outer_dim])
    ret = tf.matmul(input_tensor, w)
    if len(outer_dims) > 1:
        ret = tf.reshape(ret, batch_dims + outer_dims)
    return ret


# NLPGNN's DenseLayer3d matmul path (wala/ML#704) in miniature: rank-3 input,
# rank-3 weight [hidden, heads, head_size], one inner dim.
x = tf.ones((2, 4, 6))
w = tf.ones((6, 3, 5))
out = einsum_via_matmul(x, w, 1)

assert out.shape == (2, 4, 3, 5)
assert out.dtype == tf.float32

consume(out)
