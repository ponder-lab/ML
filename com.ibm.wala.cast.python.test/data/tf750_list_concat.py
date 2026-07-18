class SplitNote:
    def __init__(self, type, time, value, velocity):
        self.type = type
        self.time = time
        self.value = value
        self.velocity = velocity


def _divide_note(notes):
    result_array = []
    for note in notes:
        on = SplitNote("note_on", note.start, note.pitch, note.velocity)
        off = SplitNote("note_off", note.end, note.pitch, None)
        result_array += [on, off]
    return result_array


def _divide_note_tuple(notes):
    result_tuple = ()
    for note in notes:
        on = SplitNote("note_on", note.start, note.pitch, note.velocity)
        off = SplitNote("note_off", note.end, note.pitch, None)
        result_tuple += (on, off)
    return result_tuple


def encode_midi(notes):
    return _divide_note(notes)


def encode_midi_tuple(notes):
    return _divide_note_tuple(notes)


def consume(x):
    return x


consume(encode_midi([]))
consume(encode_midi_tuple([]))
