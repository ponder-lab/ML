def sink_str(n):
    pass


def sink_str_literal(n):
    pass


def sink_concatenated(n):
    pass


def sink_repeated(n):
    pass


def sink_formatted(n):
    pass


def sink_compared(n):
    pass


def sink_parsed(n):
    pass


def sink_literal(n):
    pass


sizes = [1, 2]
for i in range(3):
    text = str(i)
    assert isinstance(text, str)
    sink_str(text)
    concatenated = "h" + text
    assert concatenated == "h" + str(i)
    sink_concatenated(concatenated)
    repeated = "ab" * len(sizes)
    assert repeated == "abab"
    sink_repeated(repeated)
    formatted = "h%d" % i
    assert formatted == concatenated
    sink_formatted(formatted)
    compared = text == "1"
    assert isinstance(compared, bool)
    sink_compared(compared)
    parsed = int(text)
    assert parsed == i
    sink_parsed(parsed)
sink_str_literal(str(5))
sink_literal("h")
