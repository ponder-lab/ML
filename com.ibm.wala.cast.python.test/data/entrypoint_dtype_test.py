# wala/ML#860: a `dtype` argument whose value is a TEST FUNCTION'S PARAMETER. Under test
# entrypoints such a function is analysed as a root, so its parameters have no incoming value and
# resolve to nothing a dtype can be read from. That reached a strict resolver which raised, and
# because nothing catches it before the analysis returns, one unreadable value ended the analysis
# of the whole project rather than of that value.
#
# The module runs to completion: the test function is never called at import.
import numpy as np


def consume(x):
    pass


def test_astype_with_supplied_dtype(identifier_dtype):
    consume(np.arange(4).astype(identifier_dtype))
