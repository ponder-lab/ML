# Test helpers whose callers the call strings still join (wala/ML#1318): an argument computed from a
# parameter rather than being one, and a chain of helpers three deep.
import numpy as np


def consume_computed_index(t):
    assert t.dtype == np.int32 and t.shape == (5,)


def consume_chained_index(t):
    assert t.dtype == np.int32 and t.shape == (5,)


def to_numpy(data):
    if isinstance(data, list):
        return np.array(data)
    return data


def convert_sliced(data):
    return to_numpy(data[:])


def convert_both_sliced(x, index):
    return convert_sliced(x), convert_sliced(index)


def convert1(data):
    return to_numpy(data)


def convert2(data):
    return convert1(data)


def convert3(data):
    return convert2(data)


def convert_both_chained(x, index):
    return convert3(x), convert3(index)


features = np.zeros([5, 4], dtype=np.float32)
graph_index = np.array([0, 0, 1, 1, 1], dtype=np.int32)

_, computed_index = convert_both_sliced(features, graph_index)
consume_computed_index(computed_index)

_, chained_index = convert_both_chained(features, graph_index)
consume_chained_index(chained_index)
