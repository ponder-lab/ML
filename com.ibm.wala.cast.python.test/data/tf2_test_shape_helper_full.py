import tensorflow as tf


def f(a):
    pass


def assert_rank(tensor, expected_rank, name=None):
    excepted_rank_dict = {}
    if isinstance(expected_rank, int):
        excepted_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            excepted_rank_dict[x] = True
    actual_rank = tensor.shape.ndims
    if actual_rank not in excepted_rank_dict:
        raise (
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


# The real BERT/ALBERT `get_shape_list` vendored verbatim from NLPGNN's
# `nlpgnn/tools.py` (wala/ML#706): unlike the simplified single-return helper
# of `tf2_test_shape_helper_slice.py`, it patches `None` entries after the
# `as_list()` read (subscript writes on the returned list), has two `return
# shape` statements, and takes default parameters. The def-use walk must still
# follow the helper invoke to the `.shape.as_list()` chain rooted at the
# `tensor` parameter.
t = tf.ones((4, 5, 6))
x = tf.ones((30,))
k = 2
r = tf.reshape(x, get_shape_list(t)[-k:])
assert isinstance(r, tf.Tensor)
assert r.shape == (5, 6)
assert r.dtype == tf.float32
f(r)
