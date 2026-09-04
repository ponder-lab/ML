# Witnesses for the wala/ML#865 remainder: dtype-token spellings beyond the
# original field list resolve to their actual dtypes BY RESOLUTION rather than
# by a default that happened to coincide. The decisive arm is `np.int`, whose
# truth (int64) differs from the old unconditional float64 default, so int64
# here proves resolution happened. Removed-in-1.24 aliases are kept as the
# reported spellings; the file runs under NumPy < 1.24 with warnings.
import numpy as np


def via_np_int():
    return np.ones([2, 2], dtype=np.int)


def via_np_float():
    return np.ones([2, 2], dtype=np.float)


def via_np_long():
    return np.ones([2, 2], dtype=np.long)


def via_np_str():
    return np.zeros([2], dtype=np.str)


def via_np_str_():
    return np.zeros([2], dtype=np.str_)


def via_np_single():
    return np.ones([2, 2], dtype=np.single)


def via_np_intc():
    return np.ones([2, 2], dtype=np.intc)


def via_builtin_bool():
    return np.ones([2, 2], dtype=bool)


assert via_np_int().dtype == np.int64
assert via_np_float().dtype == np.float64
assert via_np_long().dtype == np.int64
assert via_np_str().dtype.kind == "U"
assert via_np_str_().dtype.kind == "U"
assert via_np_single().dtype == np.float32
assert via_np_intc().dtype == np.int32
assert via_builtin_bool().dtype == np.bool_
