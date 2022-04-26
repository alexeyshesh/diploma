import abc
from typing import List


class BaseDataset(abc.ABC):

    class Record:
        def __init__(self, *args, **kwargs):
            for attr in kwargs:
                self.__setattr__(attr, kwargs[attr])

    def __init__(self, filename: str or None = None):
        """
        :param filename: файл с данными
        """
        if filename:
            data = self._get_data(filename)
            self._data = [self.Record(**d) for d in data]
        self._counter = -1

    def __getitem__(self, key):
        if isinstance(key, str):
            return [doc.__getattribute__(key) for doc in self._data]

        if isinstance(key, slice):
            obj = self.__class__()
            obj._data = self._data[key]
            return obj

        return self._data[key]

    def __iter__(self):
        return self

    def __next__(self):
        self._counter += 1

        if self._counter >= len(self._data):
            self._counter = -1
            raise StopIteration

        return self._data[self._counter]

    def __len__(self):
        return len(self._data)

    @abc.abstractmethod
    def _get_data(self, filename: str) -> List[dict]:
        raise NotImplementedError
