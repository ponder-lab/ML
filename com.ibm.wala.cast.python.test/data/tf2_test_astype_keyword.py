import numpy as np


def consume_positional(a):
    pass


def consume_keyword(b):
    pass


# `astype` is a method-form operation: the array being narrowed is the RECEIVER, which the
# trampoline never passes as an argument. Both spellings of the dtype must reach the same result,
# the positional one and the keyword one.
arr = np.random.uniform(size=(2, 20))

positional = arr.astype(np.int32)
assert positional.shape == (2, 20), positional.shape
assert positional.dtype == np.int32, positional.dtype
consume_positional(positional)

keyword = arr.astype(dtype=np.int32)
assert keyword.shape == (2, 20), keyword.shape
assert keyword.dtype == np.int32, keyword.dtype
consume_keyword(keyword)
