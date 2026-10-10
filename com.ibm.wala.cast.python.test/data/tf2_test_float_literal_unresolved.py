# A Python float literal times an array whose dtype the analysis does not resolve. NumPy promotes
# an integral array to float64 and keeps a float32 one, so a dtype guessed without the operand's
# is not the result's: both results here are float64.
import json

import numpy as np


def consume_loaded(t):
    assert t.dtype == np.float64, t.dtype


def consume_mapped(t):
    assert t.dtype == np.float64, t.dtype


loaded = np.array(json.loads("[[1, 2, 3, 4], [5, 6, 7, 8]]"))
consume_loaded(loaded * 0.5)
mapped = np.array([list(map(int, "1,2,3,4".split(",")))])
consume_mapped(mapped * 0.5)


def consume_mixed(t):
    assert t.dtype == np.float64, t.dtype


def scale_mixed(a):
    consume_mixed(a * 0.5)


scale_mixed(np.arange(4))
scale_mixed(np.array(json.loads("[1, 2, 3, 4]")))
