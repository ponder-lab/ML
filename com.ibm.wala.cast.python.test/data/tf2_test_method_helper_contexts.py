# Test helpers called under a method, each called once with features and once with an index
# (wala/ML#1305).
import numpy as np


def consume_nested_index(t):
    assert t.dtype == np.int32 and t.shape == (5,)


def consume_nested_features(t):
    assert t.dtype == np.float32 and t.shape == (5, 4)


def consume_helper_index(t):
    assert t.dtype == np.int32 and t.shape == (5,)


def consume_walked_index(t):
    assert t.dtype == np.int32 and t.shape == (5,)


def consume_ticked_index(t):
    assert t.dtype == np.int32 and t.shape == (5,)


def to_numpy(data):
    if isinstance(data, list):
        return np.array(data)
    return data


def common(data):
    return to_numpy(data)


def walk(data, n):
    if n > 0:
        return walk(data, n - 1)
    return to_numpy(data)


def tick(data, n):
    if n > 0:
        return tock(data, n - 1)
    return to_numpy(data)


def tock(data, n):
    return tick(data, n)


class Graph:
    def __init__(self, x, index):
        self.x = x
        self.index = index

    def sample(self):
        def sample_common(data):
            return to_numpy(data)

        return sample_common(self.x), sample_common(self.index)


class Sampler:
    def __init__(self, x, index):
        self.x = x
        self.index = index

    def sample(self):
        x = self.x
        index = self.index
        return common(x), common(index)


class Walker:
    def __init__(self, x, index):
        self.x = x
        self.index = index

    def sample(self):
        return walk(self.x, 2), walk(self.index, 2)


features = np.zeros([5, 4], dtype=np.float32)
graph_index = np.array([0, 0, 1, 1, 1], dtype=np.int32)

nested_features, nested_index = Graph(features, graph_index).sample()
consume_nested_features(nested_features)
consume_nested_index(nested_index)

_, helper_index = Sampler(features, graph_index).sample()
consume_helper_index(helper_index)

_, walked_index = Walker(features, graph_index).sample()
consume_walked_index(walked_index)


class Ticker:
    def __init__(self, x, index):
        self.x = x
        self.index = index

    def sample(self):
        return tick(self.x, 2), tick(self.index, 2)


_, ticked_index = Ticker(features, graph_index).sample()
consume_ticked_index(ticked_index)
