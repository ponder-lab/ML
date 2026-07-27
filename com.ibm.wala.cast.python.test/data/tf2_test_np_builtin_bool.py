# The `bool` builtin as a dtype token (wala/ML#796): like `int` and `float`, the builtin
# resolves to a dtype through the front-end's builtin-function table.
import numpy as np


def consume(m):
    pass


mask = np.array([True, False, True], dtype=bool)
assert mask.dtype == np.bool_
assert mask.shape == (3,)
consume(mask)
