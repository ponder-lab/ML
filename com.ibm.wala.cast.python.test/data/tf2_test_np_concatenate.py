# Test `np.concatenate`: the arrays join along `axis`, and the dtype is NumPy's promotion of theirs,
# so an integral array beside a `float64` one gives `float64` in either order.
import numpy as np


def consume_concat(t):
    assert t.shape == (4,)
    assert t.dtype == np.float64


def consume_int_first(t):
    assert t.shape == (4,)
    assert t.dtype == np.float64


def consume_scaled(t):
    assert t.shape == (4,)
    assert t.dtype == np.float64


bbox = np.array([1, 2, 3, 4])
bbox_xywh = np.concatenate([(bbox[2:] + bbox[:2]) * 0.5, bbox[2:] - bbox[:2]], axis=-1)
consume_concat(bbox_xywh)
consume_int_first(np.concatenate([bbox[2:] - bbox[:2], (bbox[2:] + bbox[:2]) * 0.5]))
consume_scaled(1.0 * bbox_xywh)
