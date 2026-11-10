import os
import tempfile


def sink_stripped(n):
    pass


def sink_stripped_literal(n):
    pass


def sink_replaced(n):
    pass


def sink_joined(n):
    pass


def sink_upper(n):
    pass


def sink_piece(n):
    pass


def sink_line_stripped(n):
    pass


def sink_listed_line(n):
    pass


def sink_not_a_string(n):
    pass


class Trimmer:
    def strip(self):
        return 1


def file_lines_to_list(path):
    with open(path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content


path = os.path.join(tempfile.mkdtemp(), "a.txt")
with open(path, "w") as f:
    f.write(" a,b \n")

for i in range(3):
    text = str(i)
    stripped = text.strip()
    assert stripped == text
    sink_stripped(stripped)
    replaced = text.replace("1", "2")
    assert isinstance(replaced, str)
    sink_replaced(replaced)
    joined = ",".join([text, text])
    assert joined == text + "," + text
    sink_joined(joined)
    upper = text.upper()
    assert upper == text
    sink_upper(upper)
    for piece in text.split(","):
        assert piece == text
        sink_piece(piece)
stripped_literal = " h ".strip()
assert stripped_literal == "h"
sink_stripped_literal(stripped_literal)
for line in open(path).readlines():
    line_stripped = line.strip()
    assert line_stripped == "a,b"
    sink_line_stripped(line_stripped)
for listed_line in file_lines_to_list(path):
    assert listed_line == "a,b"
    sink_listed_line(listed_line)
not_a_string = Trimmer().strip()
assert not_a_string == 1
sink_not_a_string(not_a_string)
