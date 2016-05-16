execfile('simple_rnn.py')

def test_answer():
    assert text_to_dict("Costa coffee") == {"a": 0, " ": 1, "c": 2, "e": 3, "f": 4, "o": 5, "s": 6, "t": 7}
