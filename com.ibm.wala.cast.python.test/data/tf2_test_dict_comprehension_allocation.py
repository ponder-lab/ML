# Test for https://github.com/wala/ML/issues/1075.


def consume(d):
    assert isinstance(d, dict)
    return d


def built(source):
    return consume({k: v for k, v in source.items()})


def literal():
    return consume({"a": 1})


built({"a": 1})
literal()
