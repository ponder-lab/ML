# The metrics-mask shape from the corpus (wala/ML#774): `np.zeros` feeds `np.array` with an
# explicit but statically-unresolvable dtype, which overrides the source's dtype at runtime; the
# result's dtype must stay unknown rather than borrowing the operand's float64.
import numpy as np


def consume(m):
    pass


mask = np.zeros(5)
m = np.array(mask, dtype=np.bool_)
assert m.dtype == np.bool_
assert m.shape == (5,)
consume(m)
