# The factory writes library defaults that the caller overwrites (wala/ML#769).


class Holder:
    pass


def make_param():
    p = Holder()
    p.batch_size = 10
    p.maxlen = 100
    return p
