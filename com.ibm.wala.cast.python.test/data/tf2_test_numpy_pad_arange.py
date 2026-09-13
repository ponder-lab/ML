import random

import numpy as np

# `np.arange`, `np.pad` and `np.random.randint` (wala/ML#909). A pad's output extent is the input's
# extent plus its widths; when the widths are written against the input's own length, the sum is a
# constant the program fixes by cancellation, which the analysis folds as terms. Every shape below
# is asserted by the program itself.


def consume_arange_stop(a):
    pass


def consume_arange_bounds(a):
    pass


def consume_arange_step(a):
    pass


def consume_arange_keywords(a):
    pass


def consume_arange_unresolved(a):
    pass


def consume_pad_scalar(p):
    pass


def consume_pad_pair(p):
    pass


def consume_pad_axes(p):
    pass


def consume_pad_cancels(p):
    pass


def consume_pad_cancels_scaled(p):
    pass


def consume_pad_cancels_draw(p):
    pass


def consume_pad_open(p):
    pass


def consume_pad_unknown_width(p):
    pass


def consume_randint(r):
    pass


def consume_arange_mixed(a):
    pass


def consume_pad_typed_input(p):
    pass


def consume_pad_typed_unresolved(p):
    pass


def consume_pad_scaled_left(p):
    pass


def consume_pad_two_arrays(p):
    pass


def consume_pad_stepped_input(p):
    pass


def consume_pad_stop_only(p):
    pass


def consume_pad_single_width(p):
    pass


def consume_pad_single_pair(p):
    pass


def consume_pad_float_bound(p):
    pass


def consume_pad_coefficient(p):
    pass


def consume_pad_floordiv(p):
    pass


a = np.arange(10)
assert a.shape == (10,) and a.dtype == np.int64, a
consume_arange_stop(a)

b = np.arange(2, 10)
assert b.shape == (8,), b.shape
consume_arange_bounds(b)

c = np.arange(0, 10, 3)
assert c.shape == (4,), c.shape
consume_arange_step(c)

d = np.arange(start=3, stop=7)
assert d.shape == (4,), d.shape
consume_arange_keywords(d)

n = random.randint(1, 9)
m = random.randint(0, 5)

# Bounds the program decides at run time: rank one, extent unresolved.
e = np.arange(0, n)
assert e.shape == (n,), e.shape
consume_arange_unresolved(e)

p1 = np.pad(np.zeros(5), 2)
assert p1.shape == (9,), p1.shape
consume_pad_scalar(p1)

p2 = np.pad(np.ones(5), (1, 2))
assert p2.shape == (8,), p2.shape
consume_pad_pair(p2)

p3 = np.pad(np.zeros((2, 3)), ((1, 1), (0, 2)))
assert p3.shape == (4, 5), p3.shape
consume_pad_axes(p3)

# The cancellation: an unresolved length padded by `10 - n` is exactly 10 long.
p4 = np.pad(e, (0, 10 - n))
assert p4.shape == (10,), p4.shape
consume_pad_cancels(p4)

# Two cancellations through an elementwise rescale: `(m + n) - m` is `n`, then `n + (12 - n)`.
f = np.arange(start=m, stop=m + n)
p5 = np.pad(f / 2, (0, 12 - n))
assert p5.shape == (12,), p5.shape
consume_pad_cancels_scaled(p5)

# The random branch: a sized integer draw padded against its own size.
g = np.random.randint(100, size=n)
assert g.shape == (n,) and g.dtype == np.int64, g
consume_randint(g)
p6 = np.pad(g, pad_width=(0, 15 - n), mode="constant", constant_values=-1)
assert p6.shape == (15,), p6.shape
consume_pad_cancels_draw(p6)

# A width that does not cancel the unresolved length stays unresolved; the rank is still one.
p7 = np.pad(e, (0, 1))
assert p7.shape == (n + 1,), p7.shape
consume_pad_open(p7)

# A constant input with a width the program decides at run time: unresolved, rank one.
p8 = np.pad(np.zeros(5), (0, n))
assert p8.shape == (5 + n,), p8.shape
consume_pad_unknown_width(p8)

# One positional bound beside a keyword: `2` is the start when `stop` is named.
h = np.arange(2, stop=10)
assert h.shape == (8,), h.shape
consume_arange_mixed(h)

# An input the chase does not know how to read but whose shape is typed: the fold uses the shape.
p9 = np.pad(np.eye(3), 1)
assert p9.shape == (5, 5), p9.shape
consume_pad_typed_input(p9)

# The same with an unresolved typed shape: the axes stay unresolved, the rank is kept.
p10 = np.pad(np.eye(n), 1)
assert p10.shape == (n + 2, n + 2), p10.shape
consume_pad_typed_unresolved(p10)

# The scalar on the left of the elementwise operation. The product is int64 at run time; the
# analysis types an integer scalar's product with an array of unknown dtype as int32 (wala/ML#922),
# a standing elementwise rule this file does not test, so the JUnit expectation carries that dtype
# knowingly and says what to do when the rule changes.
p11 = np.pad(2 * e, (0, 10 - n))
assert p11.shape == (10,) and p11.dtype == np.int64, p11
consume_pad_scaled_left(p11)

# An elementwise operation of two arrays: the chase declines and the typed extent is unresolved.
p12 = np.pad(e + e, (0, 10 - n))
assert p12.shape == (10,), p12.shape
consume_pad_two_arrays(p12)

# An arange with a step: its length is not `stop - start`, so the chase declines.
p13 = np.pad(np.arange(0, 2 * n, 2), (0, 10 - n))
assert p13.shape == (10,), p13.shape
consume_pad_stepped_input(p13)

# An arange with only a stop: the start is zero.
p14 = np.pad(np.arange(n), (0, 10 - n))
assert p14.shape == (10,), p14.shape
consume_pad_stop_only(p14)

# A one-element width tuple pads every side.
p15 = np.pad(np.zeros(5), (2,))
assert p15.shape == (9,), p15.shape
consume_pad_single_width(p15)

# A single pair pads every axis alike.
p16 = np.pad(np.zeros((2, 3)), ((1, 2),))
assert p16.shape == (5, 6), p16.shape
consume_pad_single_pair(p16)

# A float bound is not a term the fold expresses.
p17 = np.pad(np.arange(0.5, n), (0, 10 - n))
assert p17.shape == (10,), p17.shape
consume_pad_float_bound(p17)

# Coefficients cancel too: `2n` elements padded by `20 - 2n`.
p18 = np.pad(np.arange(2 * n), (0, 20 - 2 * n))
assert p18.shape == (20,), p18.shape
consume_pad_coefficient(p18)

# An operator the fold does not express (floor division) leaves the width unresolved.
p19 = np.pad(np.zeros(5), (0, n // 2))
assert p19.shape == (5 + n // 2,), p19.shape
consume_pad_floordiv(p19)
