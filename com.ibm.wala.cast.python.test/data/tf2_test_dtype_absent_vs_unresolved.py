# wala/ML#865: an absent dtype argument and a supplied-but-unresolved one are different states and
# want opposite answers. Absence has a correct API default; a supplied argument whose value the
# analysis cannot read has no evidence at all, and answering it with the default asserts a guess.
#
# An empty points-to set cannot tell them apart, because it is empty in both cases. Presence is a
# fact about the program text, so it has to be read from the call site.

import numpy as np


def absent():
    # No dtype argument. NumPy's documented default applies and float64 is CORRECT here, not a
    # guess, so this must keep reporting it.
    return np.ones([2, 3])


def supplied_and_resolvable():
    # Supplied and readable. The argument decides.
    return np.ones([2, 3], dtype=np.int32)


def supplied_but_unresolved():
    # Supplied, but spelled with an attribute the model file carries no field for, so nothing
    # resolves. There is no evidence, and float64 here would be the defect: it is indistinguishable
    # downstream from a dtype that genuinely resolved to float64.
    return np.ones([2, 3], dtype=np.int)


def supplied_none():
    # An explicit `None` really does mean "use the default", so this one is NOT the defect and must
    # keep reporting float64.
    return np.ones([2, 3], dtype=None)


assert absent().dtype == np.float64
assert supplied_and_resolvable().dtype == np.int32
assert supplied_but_unresolved().dtype == np.int_
assert supplied_none().dtype == np.float64
