from src.datasets.base_dataset import BaseDataset

from src.models.sklearn_model import SKLearnModel


class SKVectorModel(SKLearnModel):

    __preprocess_cash = {}

    def __init__(self, classifier, vectorizer, preprocess=None, balancer=None):
        super().__init__(classifier, balancer)
        self.preprocess = preprocess or (lambda x: x)
        self.vectorizer = vectorizer

        if self.preprocess not in self.__preprocess_cash:
            self.__preprocess_cash[self.preprocess] = {}

    def get_x_train(self, dataset: BaseDataset):
        corpus = []
        for doc in dataset:
            if doc.text not in self.__preprocess_cash:
                self.__preprocess_cash[doc.text] = self.preprocess(doc.text)
            corpus.append(self.__preprocess_cash[doc.text])

        return self.vectorizer.fit_transform(corpus)

    def get_x_test(self, dataset: BaseDataset):
        corpus = []
        for doc in dataset:
            if doc.text not in self.__preprocess_cash[self.preprocess]:
                self.__preprocess_cash[self.preprocess][doc.text] = self.preprocess(doc.text)
            corpus.append(self.__preprocess_cash[self.preprocess][doc.text])

        return self.vectorizer.transform(corpus)

    def get_y_vector(self, dataset: BaseDataset):
        return [
            doc.author
            for doc in dataset
        ]

    def get_x(self, record):
        return self.vectorizer.transform(
            [
                self.preprocess(record.text)
            ]
        )[0]


class DenseSKVectorModel(SKLearnModel):

    def __init__(self, classifier, vectorizer, preprocess=None, balancer=None):
        super().__init__(classifier, balancer)
        self.preprocess = preprocess or (lambda x: x)
        self.vectorizer = vectorizer

    def get_x_train(self, dataset: BaseDataset):
        corpus = [
            self.preprocess(doc.text)
            for doc in dataset
            if len(self.preprocess(doc.text))
        ]
        return self.vectorizer.fit_transform(corpus).toarray()

    def get_x_test(self, dataset: BaseDataset):
        corpus = [
            self.preprocess(doc.text)
            for doc in dataset
            if len(self.preprocess(doc.text))
        ]
        return self.vectorizer.transform(corpus).toarray()

    def get_y_vector(self, dataset: BaseDataset):
        return [
            doc.tonality
            for doc in dataset
            if len(self.preprocess(doc.text))
        ]

    def get_x(self, record):
        return self.vectorizer.transform(
            [
                self.preprocess(record.text)
            ]
        )[0]
