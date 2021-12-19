import re
from typing import Callable, Collection, List



def get_by_ngrams(iterable: Collection, by: int = 2):
    for i in range(by, len(iterable) + 1):
        yield tuple(iterable[i-by:i])
