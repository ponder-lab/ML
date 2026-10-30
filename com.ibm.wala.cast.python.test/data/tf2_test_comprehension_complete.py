# A list a comprehension builds holds only the elements the comprehension produces, so its elements
# are all of it. A list written under a key the analysis cannot name may hold any element.


def consume_range(c):
    assert len(c) == 3 and c[0] == (0, 0, 0)


def consume_constant(c):
    assert len(c) == 3 and c[0] == (1, 2, 3)


def consume_keyed(c):
    assert len(c) == 2 and c[0] == 7


def build_keyed(k):
    l = [0, 0]
    l[k] = 7
    return l


consume_range([(i, i, i) for i in range(3)])
consume_constant([(1, 2, 3) for _ in range(3)])
consume_keyed(build_keyed(0))
