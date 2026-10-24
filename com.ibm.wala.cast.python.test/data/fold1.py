def sink(v):
    return v


a = sink(0.5 + 0.25)
b = sink(0.1 + 0.2)
c = sink(0.4 + 0.5)
d = sink(1 + 1)
e = sink(1 < 2)
f = sink(1e308 * 10)
assert a == 0.75 and b != c and d == 2 and e is True and f == float("inf")
