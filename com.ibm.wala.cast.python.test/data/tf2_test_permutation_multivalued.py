import numpy as np


def consume_multivalued(a):
    pass


# The integer argument reaches this draw with more than one value, so no single length resolves.
# The result is still a vector of SOME length; reporting the integer's own scalar shape would be a
# confidently wrong rank rather than an unresolved extent.
flag = len("ab") > 1
n = 4
if flag:
    n = 7

shuffled = np.random.permutation(n)
assert shuffled.shape == (7,), shuffled.shape
assert shuffled.dtype == np.int64, shuffled.dtype

consume_multivalued(shuffled)
