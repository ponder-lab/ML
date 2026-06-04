# Fixture mirroring `create_attention_mask_from_input_mask` (and its
# `assert_rank` / `get_shape_list` helpers) from
# `kyzhouhzau/NLPGNN/nlpgnn/tools.py`. The function builds a 3D attention mask
# from a 2D input mask, exercising `tensor.shape.as_list()`, `tf.reshape` with
# shape-list-derived dimensions, `tf.ones`, and broadcast multiplication. The
# fixture lets `TestTensorflow2Model` pin the parameter types of `from_tensor`
# and `to_mask` for tensor-type inference coverage.
import tensorflow as tf


def assert_rank(tensor, expected_rank, name=None):
    excepted_rank_dict = {}
    if isinstance(expected_rank, int):
        excepted_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            excepted_rank_dict[x] = True
    actual_rank = tensor.shape.ndims
    if actual_rank not in excepted_rank_dict:
        raise ValueError(
            "For tensor {} , the actual rank {} is not equal"
            "to expected rank {}".format(name, actual_rank, str(expected_rank))
        )


def get_shape_list(tensor, expected_rank=None, name=None):
    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

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


def create_attention_mask_from_input_mask(from_tensor, to_mask):
    assert_rank(from_tensor, [2, 3])
    from_shape = get_shape_list(from_tensor)
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]
    to_shape = get_shape_list(to_mask)
    to_seq_length = to_shape[1]

    to_mask = tf.cast(tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

    broadcast_ones = tf.ones(shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

    mask = broadcast_ones * to_mask
    return mask


# Driver: from_tensor [B, F, D] = (2, 3, 4), to_mask [B, T] = (2, 5).
from_tensor = tf.constant(
    [
        [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]],
        [[1.3, 1.4, 1.5, 1.6], [1.7, 1.8, 1.9, 2.0], [2.1, 2.2, 2.3, 2.4]],
    ],
    dtype=tf.float32,
)
to_mask = tf.constant([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=tf.int32)

assert from_tensor.shape == (2, 3, 4) and from_tensor.dtype == tf.float32
assert to_mask.shape == (2, 5) and to_mask.dtype == tf.int32

mask = create_attention_mask_from_input_mask(from_tensor, to_mask)
assert mask.shape == (2, 3, 5) and mask.dtype == tf.float32
