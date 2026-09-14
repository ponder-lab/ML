# An array built from builtin conversions, read by a reader that two callers give different
# conversions (wala/ML#925). With `dtype` the Python `int` for one caller and `float` for the other,
# the dtype argument does not decide the array's dtype, so the element walk runs and meets the
# results of calling a builtin, which are not allocations. A lookup that throws there floors the
# whole array to unknown inside the resolver's catch; a lookup that declines floors only the dtype.
import numpy as np
import tensorflow as tf


def consume_ints(a):
    pass


def consume_floats(b):
    pass


class Reader:
    def read_file(self, src, dtype=None):
        # The conversion is passed one hop further, as the field site does: the array is built by
        # the helper, whose `dtype` is a parameter of a parameter.
        return self.read_raw_text(src, dtype)

    def read_raw_text(self, src, dtype):
        rows = [[dtype(x) for x in line.split(",")] for line in src]
        return np.array(rows, dtype=dtype)


reader = Reader()
ints = reader.read_file(["1,2,3", "4,5,6"], dtype=int)
assert ints.shape == (2, 3) and ints.dtype == np.int64
consume_ints(tf.constant(ints))

floats = reader.read_file(["1.5,2.5", "3.5,4.5"], dtype=float)
assert floats.shape == (2, 2) and floats.dtype == np.float64
consume_floats(tf.constant(floats))


def consume_identity(e):
    pass


# A builtin's type token that shares a container with a tensor reaches the value readers as one
# member of the operand's points-to set, beside the tensor's allocation. A lookup that throws floors
# the whole operand to unknown; a lookup that declines contributes nothing for the token and keeps
# the tensor's shape. The runtime skips the token, which no operation accepts.
mixed = [tf.ones((2, 3)), int]
for item in mixed:
    if isinstance(item, type):
        continue
    same = tf.identity(item)
    assert same.shape == (2, 3) and same.dtype == tf.float32
    consume_identity(same)
