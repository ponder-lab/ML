# Each `tf.reshape` below runs eagerly. Under `tf.function`, the ones whose shape is a scalar
# raise "Shape must be rank 1 but is rank 0" (wala/ML#1147).
import numpy as np
import tensorflow as tf


def by_literal(x):
    return tf.reshape(x, -1)


def by_keyword(x):
    return tf.reshape(x, shape=-1)


def by_scalar_tensor(x):
    return tf.reshape(x, tf.constant(-1))


def by_numpy_scalar(x):
    return tf.reshape(x, np.int32(-1))


def by_list(x):
    return tf.reshape(x, [-1])


def by_tuple(x):
    return tf.reshape(x, (3, 4))


def by_list_of_computed(x):
    return tf.reshape(x, [x.shape[0] * x.shape[1]])


def by_shape_tensor(x, y):
    return tf.reshape(x, tf.shape(y))


def by_numpy_vector(x):
    return tf.reshape(x, np.array([-1]))


def by_computed_int(x):
    return tf.reshape(x, x.shape[0] * x.shape[1])


def by_either(x, flat):
    shape = -1 if flat else [3, 4]
    return tf.reshape(x, shape)


class Opaque:
    pass


def by_list_or_unclassified(x, flat):
    # Only the list is passed here; the other value is one the analysis cannot classify.
    shape = [-1] if flat else Opaque()
    return tf.reshape(x, shape)


x = tf.zeros([3, 4])
for f in (
    by_literal,
    by_keyword,
    by_scalar_tensor,
    by_numpy_scalar,
    by_list,
    by_list_of_computed,
    by_numpy_vector,
    by_computed_int,
):
    assert f(x).shape == (12,)
assert by_tuple(tf.zeros([12])).shape == (3, 4)
assert by_shape_tensor(tf.zeros([12]), x).shape == (3, 4)
assert by_either(x, True).shape == (12,)
assert by_either(x, False).shape == (3, 4)
assert by_list_or_unclassified(x, True).shape == (12,)

for f in (by_literal, by_keyword, by_scalar_tensor, by_numpy_scalar, by_computed_int):
    try:
        tf.function(f)(x)
        raise AssertionError(f.__name__)
    except ValueError as e:
        assert "Shape must be rank 1 but is rank 0" in str(e), e
for f in (by_list, by_list_of_computed, by_numpy_vector):
    assert tf.function(f)(x).shape == (12,)
assert tf.function(by_tuple)(tf.zeros([12])).shape == (3, 4)
assert tf.function(by_shape_tensor)(tf.zeros([12]), x).shape == (3, 4)
