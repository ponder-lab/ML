# A function the call graph does not reach may still run. Its `update` on a global it declares may
# add an element to the dict that global is written.
CACHE = {"a": 1, "b": 2}


def consume(c):
    pass


def remember(entries):
    global CACHE
    CACHE.update(entries)


assert len(CACHE) == 2 and CACHE["a"] == 1
consume(CACHE)
