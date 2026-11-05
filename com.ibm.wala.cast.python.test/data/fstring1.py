def sink_fstring(n):
    pass


def sink_plain(n):
    pass


def sink_literal(n):
    pass


def sink_escaped(n):
    pass


def sink_call(n):
    pass


def sink_concatenated(n):
    pass


def sink_spec(n):
    pass


def sink_conversion(n):
    pass


def sink_debug(n):
    pass


def sink_multiline(n):
    pass


def sink_guarded(n):
    pass


def sink_subscript(n):
    pass


def side():
    return "s"


def side_concatenated():
    return "c"


def width():
    return 4


def shown():
    return 1


def debugged():
    return 2


def side_multiline():
    return "m"


def side_guarded():
    return "g"


def table():
    return {"k": 3}


names = []
for i in range(3):
    name = f"h{i}"
    names.append(name)
    sink_fstring(name)
assert names == ["h0", "h1", "h2"]

plain = f"plain"
assert plain == "plain"
sink_plain(plain)

literal = "h"
sink_literal(literal)

escaped = f"{{h}}"
assert escaped == "{h}"
sink_escaped(escaped)

called = f"{side()}"
assert called == "s"
sink_call(called)

concatenated = "pre " f"{side_concatenated()}"
assert concatenated == "pre c"
sink_concatenated(concatenated)

spec = f"{3:{width()}}"
assert spec == "   3"
sink_spec(spec)

conversion = f"{shown()!r}"
assert conversion == "1"
sink_conversion(conversion)

debug = f"{debugged()=}"
assert debug == "debugged()=2"
sink_debug(debug)

multiline = f"""first
{side_multiline()} last"""
assert multiline == "first\nm last"
sink_multiline(multiline)

try:
    guarded = f"{side_guarded()}"
except ValueError:
    guarded = None
assert guarded == "g"
sink_guarded(guarded)

subscript = f"{table()['k']}"
assert subscript == "3"
sink_subscript(subscript)
