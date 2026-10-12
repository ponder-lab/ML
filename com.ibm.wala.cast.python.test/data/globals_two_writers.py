a = 10
b = 20


def f():
    global a
    a = a + 1


def g():
    global a, b
    a = 0
    b = b + 1


f()
assert a == 11
g()
assert a == 0
assert b == 21
