import csv
from typing import List

from src.datasets.base_dataset import BaseDataset


class CSVDataset(BaseDataset):

    @property
    def authors(self):
        return set(doc.author for doc in self._data)

    def _get_data(self, filename: str) -> List[dict]:
        data = []
        with open(filename) as f:
            input_file = csv.DictReader(f)
            for row in input_file:
                data.append(
                    {
                        'author': (
                            row['Author name']
                            if 'Author name' in row
                            else row['Author Name']
                        ),
                        'text': row['Content'],
                    },
                )
        return data
