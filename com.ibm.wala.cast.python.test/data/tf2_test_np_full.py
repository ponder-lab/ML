import numpy as np


def consume_full_int32(tensor):
    pass


def consume_full_float(tensor):
    pass


def consume_full_int(tensor):
    pass


def consume_full_bool(tensor):
    pass


def consume_full_np_scalar(tensor):
    pass


def consume_full_like(tensor):
    pass


def consume_full_like_float_fill(tensor):
    pass


def consume_full_like_shape(tensor):
    pass


def consume_concat(tensor):
    pass


# `np.full(shape, fill_value, dtype)`: the shape comes from the shape argument and
# the dtype from the explicit `dtype` argument.
a = np.full([3], 7, dtype=np.int32)
assert a.shape == (3,)
assert a.dtype == np.int32
consume_full_int32(a)

# Without a `dtype`, NumPy takes the dtype of the fill value: a Python float is
# `float64`.
b = np.full((2, 3), 1.5)
assert b.shape == (2, 3)
assert b.dtype == np.float64
consume_full_float(b)

# A Python int fill value is `int64`.
c = np.full([4], 2)
assert c.shape == (4,)
assert c.dtype == np.int64
consume_full_int(c)

# A Python bool fill value is `bool`.
d = np.full([2], True)
assert d.shape == (2,)
assert d.dtype == np.bool_
consume_full_bool(d)

# A NumPy scalar fill value keeps its own dtype.
e = np.full([2], np.float32(1.0))
assert e.shape == (2,)
assert e.dtype == np.float32
consume_full_np_scalar(e)

arr = np.zeros((2, 3), dtype=np.int32)

# `np.full_like(a, fill_value)`: the shape and the dtype are `a`'s.
f = np.full_like(arr, 0)
assert f.shape == (2, 3)
assert f.dtype == np.int32
consume_full_like(f)

# The fill value does not change `full_like`'s dtype: a float fill of an int32
# array is still int32.
g = np.full_like(arr, 0.5)
assert g.shape == (2, 3)
assert g.dtype == np.int32
consume_full_like_float_fill(g)

# An explicit `shape` overrides `a`'s shape but not its dtype.
h = np.full_like(arr, 1, shape=(4,))
assert h.shape == (4,)
assert h.dtype == np.int32
consume_full_like_shape(h)


def make_ids(n, i, m, j):
    return np.concatenate(
        [np.full([n], i, dtype=np.int32), np.full([m], j, dtype=np.int32)]
    )


k = make_ids(2, 0, 3, 1)
assert k.shape == (5,)
assert k.dtype == np.int32
consume_concat(k)
