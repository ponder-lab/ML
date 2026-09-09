# A numpy array round-tripped through a Python list and rebuilt with np.array. The reconstruction
# preserves the element kind, so a float64 source comes back float64 and an int64 source int64,
# and the rank is unchanged because the nesting depth is unchanged.
import numpy as np


def consume_float_roundtrip(a):
    pass


def consume_int_roundtrip(a):
    pass


def consume_float_source(a):
    pass


floats = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
assert floats.dtype == np.float64
assert floats.shape == (2, 2)
consume_float_source(floats)

float_rt = np.array(floats.tolist())
assert float_rt.dtype == np.float64
assert float_rt.shape == (2, 2)
consume_float_roundtrip(float_rt)

ints = np.array([[1, 2], [3, 4]], dtype=np.int64)
int_rt = np.array(ints.tolist())
assert int_rt.dtype == np.int64
assert int_rt.shape == (2, 2)
consume_int_roundtrip(int_rt)
