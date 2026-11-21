# Test https://github.com/wala/ML/issues/1308: `np.clip`, `np.expand_dims` and `np.squeeze` of a value
# that only the sidecar types. The value has no points-to members, so each function's own read of
# its argument finds nothing, and the result gets the argument's type only through a feed.
import os
import pickle
import tempfile

import numpy as np
from loader import (
    load_clip,
    load_clip_by_array,
    load_clip_by_float,
    load_clip_by_negative,
    load_clip_by_upper_array,
    load_clip_from_below,
    load_expanded,
    load_squeezed,
)


def fed_clip(x):
    assert x.shape == (4, 3) and x.dtype == np.float32


def fed_clip_by_negative(x):
    assert x.shape == (4, 3) and x.dtype == np.float32


def fed_clip_by_array(x):
    assert x.shape == (3, 4, 3) and x.dtype == np.float64


def fed_clip_by_upper_array(x):
    assert x.shape == (3, 4, 3) and x.dtype == np.float64


def fed_clip_from_below(x):
    assert x.shape == (4, 3) and x.dtype == np.float32


def fed_clip_by_float(x):
    assert x.shape == (4, 3) and x.dtype == np.float32


def fed_expanded(x):
    assert x.shape == (1, 4, 3) and x.dtype == np.float32


def fed_squeezed(x):
    assert x.shape == (4, 3) and x.dtype == np.float32


path = os.path.join(tempfile.mkdtemp(), "boxes.pickle")
with open(path, "wb") as f:
    pickle.dump(np.ones((4, 3), dtype=np.float32), f)

fed_clip(load_clip(path))
fed_clip_by_negative(load_clip_by_negative(path))
fed_clip_by_array(load_clip_by_array(path))
fed_clip_by_upper_array(load_clip_by_upper_array(path))
fed_clip_from_below(load_clip_from_below(path))
fed_clip_by_float(load_clip_by_float(path))
fed_expanded(load_expanded(path))
fed_squeezed(load_squeezed(path))
