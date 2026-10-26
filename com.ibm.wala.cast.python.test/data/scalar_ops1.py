# The result of an operator over Python scalars points to the type-level key of its type
# (wala/ML#1149); an operand of unknown kind gives no result.


class Box:
    def __add__(self, other):
        return self


def sink_counter(v):
    return v


def sink_compare(v):
    return v


def sink_scaled(v):
    return v


def sink_quotient(v):
    return v


def sink_floor(v):
    return v


def sink_negated(v):
    return v


def sink_not(v):
    return v


def sink_both(v):
    return v


def sink_unknown(v):
    return v


n = 0
for _ in range(3):
    n += 1
assert isinstance(sink_counter(n), int)
assert isinstance(sink_compare(n < 3), bool)
assert isinstance(sink_scaled(n * 0.5), float)
assert isinstance(sink_quotient(n / 2), float)
assert isinstance(sink_floor(n // 2), int)
assert isinstance(sink_negated(-n), int)
assert isinstance(sink_not(not n), bool)
assert isinstance(sink_both((n > 0) & (n < 5)), bool)
assert isinstance(sink_unknown(Box() + 1), Box)
