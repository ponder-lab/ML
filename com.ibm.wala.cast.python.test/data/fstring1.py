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


def sink_operators(n):
    pass


def sink_nested(n):
    pass


def compared():
    return 1


def unequal():
    return 1


def bounded():
    return 1


def keyed():
    return 5


def nested():
    return "n"


operators = f"{compared() == 1}{unequal() != 1}{bounded() <= 1}{ {'k': keyed()}['k'] }"
assert operators == "TrueFalseTrue5"
sink_operators(operators)

inner = f"{f'{nested()}'}"
assert inner == "n"
sink_nested(inner)


def sink_raw_joined(n):
    pass


def sink_triple_quoted(n):
    pass


def sink_unparsed(n):
    pass


def side_raw():
    return "r"


def side_triple():
    return "t"


def side_unparsed():
    return "u"


raw_joined = r"\d" f"{side_raw()}"
assert raw_joined == "\\dr"
sink_raw_joined(raw_joined)

triple_quoted = f"{'''t''' + side_triple()}"
assert triple_quoted == "tt"
sink_triple_quoted(triple_quoted)

# The front end's grammar has no `:=`, so this field does not parse there.
unparsed = f"{(walrus := side_unparsed())}"
assert unparsed == "u" and walrus == "u"
sink_unparsed(unparsed)
