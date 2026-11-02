# A function the call graph does not reach may still run. Its `extend` on what a function of the
# enclosing frame returns may add an element to any list.


def consume(c):
    pass


def make_collector(factory):
    def collect(items):
        factory().extend(items)

    return collect


c = [1, 2]
assert len(c) == 2 and c[0] == 1
consume(c)
