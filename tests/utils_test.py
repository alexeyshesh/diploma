from src.utils import (
    get_by_ngrams,
    max_key,
    get_pairs,
)


def test_get_by_ngrams():
    data = [1, 2, 3, 4]
    expected_result_1 = [(1, 2), (2, 3), (3, 4)]
    expected_result_2 = [(1, 2, 3), (2, 3, 4)]

    assert list(get_by_ngrams(data)) == expected_result_1
    assert list(get_by_ngrams(data, 3)) == expected_result_2


def test_max_key():
    d = {
        'a': 2,
        'b': 3,
        'c': 1,
    }

    assert max_key(d) == 'b'


def test_get_pairs():
    s = {1, 2, 3}

    assert set(get_pairs(s)) == {
        (1, 1), (1, 2), (1, 3),
        (2, 1), (2, 2), (2, 3),
        (3, 1), (3, 2), (3, 3),
    }
