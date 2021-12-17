import csv
from typing import List


class Dataset:

    class Record:
        def __init__(self, author: str, text: str):
            self.author = author
            self.text = text

            self.authors = set()

    def __init__(self, data: List[dict]):
        """
        :param data: список словарец вида {'author': '<name>', 'text': '<text>'}
        """
        self._data = data
        self._counter = -1

    def __getitem__(self, key):
        return self.Record(
            author=self._data[key]['author'],
            text=self._data[key]['text'],
        )

    def __iter__(self):
        return self

    def __next__(self):
        self._counter += 1

        if self._counter >= len(self._data):
            self._counter = -1
            raise StopIteration

        return self.Record(
            author=self._data[self._counter]['author'],
            text=self._data[self._counter]['text'],
        )

    def __len__(self):
        return len(self._data)

    @classmethod
    def from_csv(cls, filename: str, author_col_name: str, text_col_name: str):
        data = []
        authors = set()
        with open(filename) as f:
            input_file = csv.DictReader(f)
            for row in input_file:
                data.append(
                    {
                        'author': row[author_col_name],
                        'text': row[text_col_name],
                    },
                )
                authors.add(row[author_col_name])

        obj = cls(data)
        obj.authors = authors

        return obj
