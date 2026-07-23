# Probe for wala/ML#745: two modules each define `make` with a different integer default.
# Distinct result shapes detect any cross-module union of the defaults globals.
import mod_a
import mod_b


def consume(t):
    pass


def consume2(t):
    pass


a = mod_a.make()
b = mod_b.make()
consume(a)
consume2(b)

assert a.shape == (4, 2)
assert b.shape == (5, 3)
