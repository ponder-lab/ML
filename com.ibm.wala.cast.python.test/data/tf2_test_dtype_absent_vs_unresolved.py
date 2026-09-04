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
    # Supplied, but spelled with a dtype the model file carries no field for, so nothing resolves.
    # There is no evidence, and float64 here would be the defect: it is indistinguishable
    # downstream from a dtype that genuinely resolved to float64.
    #
    # `np.int16` is used because it is a current scalar type with no `DType` yet, so it stays
    # unrepresentable BY DESIGN. The originally-reported `np.int` became RESOLVABLE when the
    # wala/ML#865 field list grew; this arm needs a spelling that is genuinely unreadable, and
    # int16's absence is the enforced one until the `DType` enum is extended.
    return np.ones([2, 3], dtype=np.int16)


def supplied_positionally():
    # Supplied positionally rather than by keyword (`shape` is positional 0, `dtype` positional 1),
    # exercising the positional half of the syntactic-presence read: a wrong offset there would
    # make a positionally supplied dtype look determinately absent and take the API default.
    return np.ones([2, 3], np.int32)


def supplied_through_kwargs(kw):
    # A call-site `**` spread could carry a dtype the call shape cannot show, so presence is
    # INDETERMINATE and the default must not be asserted. (At runtime this particular spread DOES
    # carry the dtype, so the runtime truth is int32; the analysis honestly reports unknown.)
    return np.ones([2, 3], **kw)


def supplied_none():
    # An explicit `None` really does mean "use the default", so this one is NOT the defect and must
    # keep reporting float64.
    return np.ones([2, 3], dtype=None)


def supplied_through_star(args):
    # A starred unpack could carry a dtype the call shape cannot show, and positional alignment
    # past a star is unreliable, so presence is INDETERMINATE. The default must not be asserted on
    # a call shape that cannot show whether the argument is there; only determinate absence earns
    # it. (At runtime this particular unpack carries no dtype, so float64 is what actually runs.)
    return np.ones(*args)


assert absent().dtype == np.float64
assert supplied_and_resolvable().dtype == np.int32
assert supplied_but_unresolved().dtype == np.int16
assert supplied_none().dtype == np.float64
assert supplied_through_star(([2, 3],)).dtype == np.float64
assert supplied_positionally().dtype == np.int32
assert supplied_through_kwargs({"dtype": np.int32}).dtype == np.int32
