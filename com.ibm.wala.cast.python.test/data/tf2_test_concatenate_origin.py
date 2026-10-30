# Test that the results of `np.vstack` and `np.concatenate` carry the NumPy origin, and that a value
# a function computes from them carries it to the call site.
import numpy as np


def load():
    return np.array([[0.0, 0.0, 2.0, 3.0], [1.0, 1.0, 4.0, 5.0]])


def wh(boxes):
    return boxes[..., 2:4] - boxes[..., 0:2]


def consume_vstack(x):
    assert isinstance(x, np.ndarray) and x.shape == (4, 4)


def consume_concatenate(x):
    assert isinstance(x, np.ndarray) and x.shape == (4, 4)


def consume_vstack_wh(x):
    assert isinstance(x, np.ndarray) and x.shape == (4, 2)


def consume_array_wh(x):
    assert isinstance(x, np.ndarray) and x.shape == (2, 2)


stacked = np.vstack([load() for _ in range(2)])
consume_vstack(stacked)
consume_concatenate(np.concatenate([load(), load()]))
consume_vstack_wh(wh(stacked))
consume_array_wh(wh(load()))
