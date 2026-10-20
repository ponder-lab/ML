# Test for wala/ML#1131: the `iter` and `next` calls the front end makes while lowering a `for` loop
# or a comprehension's generator record that origin, which an `iter(x)` or `next(it)` in the
# source lacks.


def iter_param(x):
    return next(iter(x)) * 2.0


def iterate_param(x):
    total = 0.0
    for e in x:
        total = total + e
    return total


def list_comp(x):
    return [e * 2.0 for e in x]


def set_comp(x):
    return {e * 2.0 for e in x}


def dict_comp(x):
    return {e: e * 2.0 for e in x}


def gen_exp(x):
    return sum(e * 2.0 for e in x)


values = [1.0, 2.0, 3.0]
assert iter_param(values) == 2.0
assert iterate_param(values) == 6.0
assert list_comp(values) == [2.0, 4.0, 6.0]
assert set_comp(values) == {2.0, 4.0, 6.0}
assert dict_comp(values) == {1.0: 2.0, 2.0: 4.0, 3.0: 6.0}
assert gen_exp(values) == 12.0
