# Test for wala/ML#1133: each iteration site records whether the source wrote a `for` statement or a
# comprehension, and which values it iterates.


def for_statement(x):
    total = 0.0
    for e in x:
        total = total + e
    return total


def dict_comp(x):
    return {e: e * 2.0 for e in x}


def same_variable(x):
    total = 0.0
    for e in x:
        total = total + e
    doubled = {e: e * 2.0 for e in x}
    return total, doubled


def nested(x, y):
    total = 0.0
    for a in x:
        for b in y:
            total = total + a * b
    return total


def for_over_dict_comp(x):
    total = 0.0
    for k in {e: e * 2.0 for e in x}:
        total = total + k
    return total


def list_comp(x, y):
    return [a * b for a in x for b in y]


def set_comp(x):
    return {e * 2.0 for e in x}


def gen_exp(x):
    return sum(e * 2.0 for e in x)


def explicit_iter(x):
    return next(iter(x)) * 2.0


values = [1.0, 2.0, 3.0]
assert for_statement(values) == 6.0
assert dict_comp(values) == {1.0: 2.0, 2.0: 4.0, 3.0: 6.0}
assert same_variable(values) == (6.0, {1.0: 2.0, 2.0: 4.0, 3.0: 6.0})
assert nested(values, values) == 36.0
assert for_over_dict_comp(values) == 6.0
assert list_comp(values, [1.0]) == [1.0, 2.0, 3.0]
assert set_comp(values) == {2.0, 4.0, 6.0}
assert gen_exp(values) == 12.0
assert explicit_iter(values) == 2.0
