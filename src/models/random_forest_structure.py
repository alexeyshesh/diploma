from nltk import sent_tokenize
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from src.deptree import build_high_level_structure
from src.ds import Dataset
from src.model import AAModel
from src.utils import get_pairs, get_by_ngrams


class RandomForestStructureModel(AAModel):

    def __init__(self, n_estimators: int = 1000):
        self._keys = {'^', 't', 'rt', 'p', 'pt', '$'}
        self.pairs = list(get_pairs(self._keys))
        self.keys = {
            self.pairs[i]: i
            for i in range(len(self.pairs))
        }
        # self.classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
        self.classifier = KNeighborsClassifier(3)

    def build_vector_from_text(self, text: str) -> list:
        vector = [1 for _ in self.keys]
        sents = sent_tokenize(text)
        for sent in sents:
            struct = ['^', *build_high_level_structure(sent), '$']
            for pair in get_by_ngrams(struct):
                vector[self.keys[pair]] += 1

        prob_vector = [0 for _ in self.keys]
        for i in range(len(vector)):
            prob_vector[i] = vector[i] / sum(
                [
                    vector[self.keys[pair]]
                    for pair in self.keys
                    if pair[0] == self.pairs[i][0]
                ]
            )

        return prob_vector

    def train(self, dataset: Dataset):
        x_train = []
        y_train = []

        for record in tqdm(dataset):
            x_train.append(self.build_vector_from_text(record.text))
            y_train.append(record.author)

        self.classifier.fit(x_train, y_train)

    def predict(self, text: str) -> str:
        return self.classifier.predict(
            [
                self.build_vector_from_text(text),
            ],
        )[0]
