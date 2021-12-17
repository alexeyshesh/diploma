import abc

from src.ds import Dataset


class AAModel(abc.ABC):
    @abc.abstractmethod
    def train(self, dataset: Dataset):
        pass

    @abc.abstractmethod
    def predict(self, text: str) -> str:
        pass
