import math

from nltk import sent_tokenize
from tqdm import tqdm

from src.deptree import build_high_level_structure
from src.ds import Dataset
from src.model import AAModel
from src.utils import get_by_ngrams, max_key


class MarkovChainModel(AAModel):

    def __init__(self):
        self.keys = {'^', 't', 'rt', 'p', 'pt', '$'}
        self.dataset: Dataset or None = None
        self.matrix = None

    def create_author_matrix(self):
        self.matrix = {}

        for author in self.dataset.authors:
            self.matrix[author] = {}
            for k1 in self.keys:
                self.matrix[author][k1] = {}
                for k2 in self.keys:
                    self.matrix[author][k1][k2] = 1  # сглаживание

    def normalize_author_matrix(self):
        for author in self.dataset.authors:
            for k in self.keys:
                div = sum(self.matrix[author][k].values())
                self.matrix[author][k] = {
                    _k: math.log(self.matrix[author][k][_k] / div)
                    for _k in self.keys
                }

    def train(self, dataset: Dataset):
        self.dataset = dataset
        self.create_author_matrix()

        for record in tqdm(dataset):
            sents = sent_tokenize(record.text)
            for sent in sents:
                struct = ['^', *build_high_level_structure(sent), '$']
                for pair in get_by_ngrams(struct):
                    self.matrix[record.author][pair[0]][pair[1]] += 1

        self.normalize_author_matrix()

    def get_author_probability(self, author: str, text: str) -> int:
        result = 0
        sents = sent_tokenize(text)
        for sent in sents:
            struct = ['^', *build_high_level_structure(sent), '$']
            for pair in get_by_ngrams(struct):
                result += self.matrix[author][pair[0]][pair[1]]

        return result

    def predict(self, text: str) -> str:
        probs = {}
        for author in self.dataset.authors:
            probs[author] = self.get_author_probability(author, text)

        return max_key(probs)
