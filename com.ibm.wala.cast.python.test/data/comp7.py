def f1(a):
    return lambda: a + 1


def f2(a):
    return lambda: a + 2


def f3(a):
    return lambda: a + 3


fs = [f1, f2, None, f3]

# Two `if` clauses on one comprehension: each is its own filter function (wala/ML#917).
vs = [f(0) for f in fs if f is not None if f is not f2]


class Holder:
    # A filtered comprehension as a method's default value is built in the class's own scope: its
    # filter is a function of its own, not a method of Holder.
    def ws(self, xs=[f(0) for f in fs if f is not None]):
        return xs


assert [f() for f in vs] == [1, 3]
assert [f() for f in Holder().ws()] == [1, 2, 3]
print([f() for f in vs])
print([f() for f in Holder().ws()])
