# A function the call graph does not reach may still run. Its `extend` on what a parameter returns
# may add an element to any list.


def consume(c):
    pass


def fill(make, items):
    make().extend(items)


c = [1, 2]
assert len(c) == 2 and c[0] == 1
consume(c)
