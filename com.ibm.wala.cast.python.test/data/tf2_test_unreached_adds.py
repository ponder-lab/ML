# A function the call graph does not reach may still run, so a call there that may add an element
# to a container it cannot name may add one to any container. This one cannot: it adds to a set the
# function builds itself. The container consumed is a dict, which an `add` may reach, since no list
# has the method (https://github.com/wala/ML/issues/1328).


def consume(c):
    assert len(c) == 2 and c["a"] == 1


def build_vocabulary(tokens):
    words = set()
    for token in tokens:
        words.add(token)
    return words


consume({"a": 1, "b": 2})
