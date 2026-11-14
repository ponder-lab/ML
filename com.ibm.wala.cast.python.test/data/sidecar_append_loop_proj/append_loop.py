# A value typed only by the project's sidecar, appended to with `np.append` in several forms
# (wala/ML#1264). Each form's result is a rank-1 int64 array, as the unpickled list of Python
# integers is.
import os
import pickle
import tempfile

import numpy as np

pad_token = 388
token_eos = 390


def write_sequence():
    path = os.path.join(tempfile.mkdtemp(), "seq.pickle")
    with open(path, "wb") as f:
        pickle.dump(np.array([1, 2, 3]), f)
    return path


def append_then_loop(fname, max_length):
    with open(fname, "rb") as f:
        data = pickle.load(f)
    data = np.append(data, token_eos)
    while len(data) < max_length:
        data = np.append(data, pad_token)
    return data


def append_twice(fname):
    with open(fname, "rb") as f:
        data = pickle.load(f)
    data = np.append(data, token_eos)
    data = np.append(data, pad_token)
    return data


def loop_only(fname, max_length):
    with open(fname, "rb") as f:
        data = pickle.load(f)
    while len(data) < max_length:
        data = np.append(data, pad_token)
    return data


def consume_append_then_loop(x):
    assert x.dtype == np.int64 and x.ndim == 1


def consume_append_twice(x):
    assert x.dtype == np.int64 and x.ndim == 1


def consume_loop_only(x):
    assert x.dtype == np.int64 and x.ndim == 1


path = write_sequence()
consume_append_then_loop(append_then_loop(path, 10))
consume_append_twice(append_twice(path))
consume_loop_only(loop_only(path, 10))
