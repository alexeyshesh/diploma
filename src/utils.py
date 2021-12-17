from typing import Collection, Set


def get_by_ngrams(iterable: Collection, by: int = 2):
    for i in range(by, len(iterable) + 1):
        yield tuple(iterable[i-by:i])


def get_pairs(some_set: Set):
    for a in some_set:
        for b in some_set:
            yield a, b


def max_key(dictionary: dict):
    return max(dictionary, key=dictionary.get)
