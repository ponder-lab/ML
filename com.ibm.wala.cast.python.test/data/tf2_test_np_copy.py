# `np.copy` returns a fresh array of its argument's dtype and shape, so a copy's rows carry the
# dtype through their slices and arithmetic as the original's do.
import numpy as np


def consume_copy(t):
    assert t.shape == (2, 5), t.shape
    assert t.dtype == np.int64, t.dtype


def consume_loop(t):
    assert t.shape == (4,), t.shape
    assert t.dtype == np.float64, t.dtype


def xywh(bc):
    return np.concatenate([(bc[2:] + bc[:2]) * 0.5, bc[2:] - bc[:2]], axis=-1)


def prepare(boxes):
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * 2
    return boxes


bbs = np.array([[10, 20, 30, 40, 1], [5, 6, 7, 8, 2]])
consume_copy(np.copy(bbs))

for bb in prepare(np.copy(bbs)):
    consume_loop(1.0 * xywh(bb[:4]))
