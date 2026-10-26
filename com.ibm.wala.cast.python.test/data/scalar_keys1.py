# A Python scalar whose value is not a literal points to one type-level key per type
# (wala/ML#1148); a literal keeps its constant key.


class Vector:
    def __init__(self, n):
        self.n = n

    def __add__(self, other):
        return Vector(self.n + (other.n if isinstance(other, Vector) else other))

    __radd__ = __add__


def sink_len(v):
    return v


def sink_int(v):
    return v


def sink_float(v):
    return v


def sink_bool(v):
    return v


def sink_range(v):
    return v


def sink_enumerate(v):
    return v


def sink_literal(v):
    return v


def sink_sum(v):
    return v


def sink_isinstance(v):
    return v


def sink_round(v):
    return v


def sink_round_digits(v):
    return v


def sink_sum_floats(v):
    return v


def sink_sum_objects(v):
    return v


def sink_round_keyword(v):
    return v


def sink_sum_start(v):
    return v


def sink_sum_start_keyword(v):
    return v


assert sink_len(len([1, 2])) == 2
assert sink_int(int("3")) == 3
assert sink_float(float("0.5")) == 0.5
assert sink_bool(bool([])) is False
for i in range(3):
    assert isinstance(sink_range(i), int)
for j, e in enumerate(["a"]):
    assert isinstance(sink_enumerate(j), int)
assert sink_literal(3) == 3
# A sum's and a round's type follows their arguments.
assert sink_sum(sum([1, 2])) == 3
assert sink_sum_floats(sum([0.5, 0.25])) == 0.75
assert isinstance(sink_sum_objects(sum([Vector(1), Vector(2)])), Vector)
assert sink_isinstance(isinstance(3, int)) is True
assert sink_round(round(float("2.5"))) == 2
assert sink_round_digits(round(float("2.25"), 1)) == 2.2
assert sink_round_keyword(round(float("2.25"), ndigits=1)) == 2.2
assert sink_sum_start(sum([1, 2], 0.5)) == 3.5
assert sink_sum_start_keyword(sum([1, 2], start=0.5)) == 3.5
