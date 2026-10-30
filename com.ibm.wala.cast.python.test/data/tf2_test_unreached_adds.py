# A function the call graph does not reach may still run, so a call there that may add an element
# to a container it cannot name may add one to any container. This one cannot: it adds to a set the
# function builds itself.


def consume(c):
    assert len(c) == 2 and c[0] == 1


def build_vocabulary(tokens):
    words = set()
    for token in tokens:
        words.add(token)
    return words


consume([1, 2])
