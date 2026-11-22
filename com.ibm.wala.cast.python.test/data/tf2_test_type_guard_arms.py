# A helper that converts its argument only when the argument has some other type, so the type test
# guarding each conversion decides which arm runs (wala/ML#1053).
import pickle

import numpy as np
import tensorflow as tf


def consume_float32_kept(x):
    pass


def consume_float64_converted(x):
    pass


def consume_float32_unequal(x):
    pass


def consume_tensor_kept(x):
    pass


def consume_tensor_mixed_token(x):
    pass


def consume_array_kept(x):
    pass


def consume_list_converted(x):
    pass


def consume_cast_x_kept(x):
    pass


def consume_cast_x_converted(x):
    pass


def consume_identity_kept(x):
    pass


def consume_cast_x_chained(x):
    pass


def consume_truth_kept(x):
    pass


def consume_string_kept(x):
    pass


def consume_string_left_kept(x):
    pass


def consume_ordered_kept(x):
    pass


def consume_subclass_converted(x):
    pass


def consume_either_converted(x):
    pass


def consume_loop_kept(x):
    pass


def consume_arms_disagree(x):
    pass


def consume_one_armed_kept(x):
    pass


def consume_arms_agree(x):
    pass


def consume_unknown_call_kept(x):
    pass


def consume_unknown_object_kept(x):
    pass


def consume_unpickled_converted(x):
    pass


def to_float32(x):
    if x.dtype == np.float64:
        x = x.astype(np.float32)
    return x


def to_int32(x):
    # A conversion to another dtype than the argument's, so its arm's value is seen if it is kept.
    if x.dtype == np.float64:
        x = x.astype(np.int32)
    return x


def float32_kept():
    x = to_int32(np.zeros((2, 3), dtype=np.float32))
    assert x.dtype == np.float32
    consume_float32_kept(x)


def float64_converted():
    x = to_float32(np.zeros((2, 3)))
    assert x.dtype == np.float32
    consume_float64_converted(x)


def to_float32_unequal(x):
    if x.dtype != np.float64:
        return x
    return x.astype(np.int32)


def float32_unequal():
    x = to_float32_unequal(np.zeros((2, 3), dtype=np.float32))
    assert x.dtype == np.float32
    consume_float32_unequal(x)


def cast_tensor(x):
    if x.dtype == tf.float64:
        x = tf.cast(x, tf.float32)
    return x


def cast_tensor_int32(x):
    if x.dtype == tf.float64:
        x = tf.cast(x, tf.int32)
    return x


def tensor_kept():
    x = cast_tensor_int32(tf.ones([2, 3]))
    assert x.dtype == tf.float32
    consume_tensor_kept(x)


def tensor_mixed_token():
    # A NumPy array's dtype equals the TensorFlow token of the same type.
    x = cast_tensor(np.zeros((2, 3)))
    assert x.dtype == tf.float32
    consume_tensor_mixed_token(x)


def from_list(x):
    if isinstance(x, list):
        x = np.array(x, dtype=np.int32)
    return x


def array_kept():
    x = from_list(np.zeros((2, 3), dtype=np.float32))
    assert x.dtype == np.float32
    consume_array_kept(x)


def list_converted():
    x = from_list([1, 2, 3])
    assert x.dtype == np.int32
    consume_list_converted(x)


def cast_x(x):
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(x, np.ndarray) and x.dtype == np.float64:
        x = x.astype(np.float32)
    return x


def cast_x_kept():
    x = cast_x(np.zeros((2, 3), dtype=np.float32))
    assert x.dtype == np.float32
    consume_cast_x_kept(x)


def cast_x_converted():
    x = cast_x(np.zeros((2, 3)))
    assert x.dtype == np.float32
    consume_cast_x_converted(x)


def cast_x_int(x):
    # The issue's helper with conversions to other dtypes than the argument's, so a kept arm's value
    # is seen: the second guard reads `x` after the first, through the first's merge.
    if isinstance(x, list):
        x = np.array(x, dtype=np.int64)
    if isinstance(x, np.ndarray) and x.dtype == np.float64:
        x = x.astype(np.int32)
    return x


def cast_x_chained():
    x = cast_x_int(np.zeros((2, 3), dtype=np.float32))
    assert x.dtype == np.float32
    consume_cast_x_chained(x)


def to_float32_identity(x):
    # `is` compares a dtype with the type object, which is never the same object.
    if x.dtype is np.float64:
        x = x.astype(np.float32)
    return x


def identity_kept():
    x = to_float32_identity(np.zeros((2, 3)))
    assert x.dtype == np.float64
    consume_identity_kept(x)


def kept_if_dtype(x):
    # Every NumPy and TensorFlow dtype is true.
    if x.dtype:
        return x
    return x.astype(np.int32)


def truth_kept():
    x = kept_if_dtype(np.zeros((2, 3), dtype=np.float32))
    assert x.dtype == np.float32
    consume_truth_kept(x)


def kept_if_named(x):
    # A dtype equals the string that names it.
    if x.dtype == "float32":
        return x
    return x.astype(np.int32)


def string_kept():
    x = kept_if_named(np.zeros((2, 3), dtype=np.float32))
    assert x.dtype == np.float32
    consume_string_kept(x)


def kept_if_named_left(x):
    if "float32" == x.dtype:
        return x
    return x.astype(np.int32)


def string_left_kept():
    x = kept_if_named_left(np.zeros((2, 3), dtype=np.float32))
    assert x.dtype == np.float32
    consume_string_left_kept(x)


def kept_if_narrower(x):
    # NumPy orders dtypes by safe casting: float32 casts safely to float64.
    if x.dtype < np.float64:
        return x
    return x.astype(np.int32)


def ordered_kept():
    x = kept_if_narrower(np.zeros((2, 3), dtype=np.float32))
    assert x.dtype == np.float32
    consume_ordered_kept(x)


class IntList(list):
    pass


def subclass_converted():
    # An instance of a program class that extends `list` is a list.
    x = from_list(IntList([1, 2, 3]))
    assert x.dtype == np.int32 and x.shape == (3,)
    consume_subclass_converted(x)


def either_converted():
    # Either a list or an array, so the test holds for one object and fails for the other.
    either = [1, 2, 3] if np.random.rand() < 2 else np.zeros((2, 3), dtype=np.float32)
    x = from_list(either)
    assert x.dtype == np.int32
    consume_either_converted(x)


def unwrap(x):
    # The loop's guard reads `x` through the merge at the loop's head, whose arm from the loop's
    # body is reached only through the guard, and the body's merge reads it again.
    while isinstance(x, list):
        if isinstance(x, list):
            x = tuple(x)
    return x


def loop_kept():
    x = unwrap(np.zeros((2, 3), dtype=np.float32))
    assert x.dtype == np.float32
    consume_loop_kept(x)


def merged_disagree():
    # The merge's arms have different dtypes, so its dtype is not one.
    y = (
        np.zeros((2, 3), dtype=np.float32)
        if np.random.rand() > 2
        else np.zeros((2, 3), dtype=np.float64)
    )
    if y.dtype == np.float64:
        return y.astype(np.int32)
    return y


def arms_disagree():
    x = merged_disagree()
    assert x.dtype == np.int32
    consume_arms_disagree(x)


def merged_agree():
    # The merge's arms have one dtype.
    y = (
        np.zeros((2, 3), dtype=np.float32)
        if np.random.rand() > 2
        else np.ones((2, 3), dtype=np.float32)
    )
    if y.dtype == np.float64:
        return y.astype(np.int32)
    return y


def arms_agree():
    x = merged_agree()
    assert x.dtype == np.float32
    consume_arms_agree(x)


def kept_if_pickled(x):
    # A call no summary or program function models.
    if pickle.dumps(x, 4):
        return x
    return x.astype(np.int32)


def unknown_call_kept():
    x = kept_if_pickled(np.zeros((2, 3), dtype=np.float32))
    assert x.dtype == np.float32
    consume_unknown_call_kept(x)


def kept_unless_list(x):
    # The value of a call no summary models is no object the analysis holds.
    y = pickle.loads(pickle.dumps(x))
    if isinstance(y, list):
        return x.astype(np.int32)
    return x


def unknown_object_kept():
    x = kept_unless_list(np.zeros((2, 3), dtype=np.float32))
    assert x.dtype == np.float32
    consume_unknown_object_kept(x)


def from_list_or_unpickled(x):
    # One arm is the value of a call nothing models, which is a list at run time.
    y = (
        pickle.loads(pickle.dumps([[1, 2, 3], [4, 5, 6]]))
        if np.random.rand() < 2
        else x
    )
    if isinstance(y, list):
        return np.array(y, dtype=np.int32)
    return y


def unpickled_converted():
    x = from_list_or_unpickled(np.zeros((2, 3), dtype=np.float32))
    assert x.dtype == np.int32 and x.shape == (2, 3)
    consume_unpickled_converted(x)


def one_armed(x):
    # `y` is defined on one arm alone.
    if np.random.rand() < 2:
        y = x
    if isinstance(y, list) or y.dtype == np.float64:
        return y.astype(np.int32)
    return y


def one_armed_kept():
    x = one_armed(np.zeros((2, 3), dtype=np.float32))
    assert x.dtype == np.float32
    consume_one_armed_kept(x)


float32_kept()
float64_converted()
float32_unequal()
tensor_kept()
tensor_mixed_token()
array_kept()
list_converted()
cast_x_kept()
cast_x_converted()
cast_x_chained()
identity_kept()
truth_kept()
string_kept()
string_left_kept()
ordered_kept()
subclass_converted()
either_converted()
loop_kept()
arms_disagree()
one_armed_kept()
arms_agree()
unknown_call_kept()
unknown_object_kept()
unpickled_converted()
