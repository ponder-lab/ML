# A Python scalar whose value is not a literal points to one type-level key per type
# (wala/ML#1148); a literal keeps its constant key.


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


assert sink_len(len([1, 2])) == 2
assert sink_int(int("3")) == 3
assert sink_float(float("0.5")) == 0.5
assert sink_bool(bool([])) is False
for i in range(3):
    assert isinstance(sink_range(i), int)
for j, e in enumerate(["a"]):
    assert isinstance(sink_enumerate(j), int)
assert sink_literal(3) == 3
# A sum's type follows its elements, so `sum` gives no scalar key.
assert sink_sum(sum([1, 2])) == 3
