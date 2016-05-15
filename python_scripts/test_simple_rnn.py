execfile('simple_rnn.py')

def test_answer():
    assert text_to_dict("Costa coffee") == {"a": 0, "c": 1, "e": 2, "f": 3, "o": 4, "s": 5, "t": 6}
