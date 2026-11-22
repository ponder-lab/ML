# Test a helper called inside another helper, which is called once with features and once with an index.
import numpy as np


def consume_index(t):
    assert t.dtype == np.int32 and t.shape == (5,)


def consume_features(t):
    assert t.dtype == np.float32 and t.shape == (5, 4)


def consume_direct_index(t):
    assert t.dtype == np.int32 and t.shape == (5,)


def to_numpy(data):
    if isinstance(data, list):
        return np.array(data)
    return data


def convert(data):
    return to_numpy(data)


def convert_both(x, index):
    return convert(x), convert(index)


def convert_both_directly(x, index):
    return to_numpy(x), to_numpy(index)


features = np.zeros([5, 4], dtype=np.float32)
graph_index = np.array([0, 0, 1, 1, 1], dtype=np.int32)

converted_features, converted_index = convert_both(features, graph_index)
consume_features(converted_features)
consume_index(converted_index)

_, direct_index = convert_both_directly(features, graph_index)
consume_direct_index(direct_index)


def consume_copied_index(t):
    assert t.dtype == np.int32 and t.shape == (5,)


def copy(data):
    return np.array(data)


def copy_through(data):
    return copy(data)


def copy_both(x, index):
    return copy_through(x), copy_through(index)


_, copied_index = copy_both(features, graph_index)
consume_copied_index(copied_index)
