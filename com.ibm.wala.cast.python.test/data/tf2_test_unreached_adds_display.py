# A function the call graph does not reach may still run, so a call there that may add an element
# to a container it cannot name may add one to any container. This one cannot: it adds to a dict
# display the function builds itself.


def consume(c):
    pass


def tally(tokens):
    counts = {}
    counts.update(tokens)
    return counts


c = [1, 2]
assert len(c) == 2 and c[0] == 1
consume(c)
