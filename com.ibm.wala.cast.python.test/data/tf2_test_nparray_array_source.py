import numpy as np


def f(t):
    return t


# The source is itself an `np.ndarray`, not a Python literal. numpy preserves the source's dtype
# rather than promoting, which the static analysis does not model, so it floors to ⊤ (the sound
# result). The nested-array shape does not propagate through the outer `np.array` either, so the
# static result is ⊤ on both axes. See wala/ML#626.
x = np.array(np.array([1, 2]))
assert x.shape == (2,)
assert x.dtype == np.int64
f(x)
