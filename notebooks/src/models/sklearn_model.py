import abc
from logging import getLogger
from typing import Any

from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV

from src.datasets.base_dataset import BaseDataset

from src.models.base_model import BaseModel

logger = getLogger(__name__)
PARAMS_GRID = {
    'KNeighborsClassifier': {
        'n_neighbors': [3, 4, 5, 6],
        'weights': ['uniform', 'distance'],
    },
    'SVC': [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ],
    'RandomForestClassifier': {
        'n_estimators': [100, 500, 1000, 2000],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2'],
        'class_weight': [None, 'balanced', 'balanced_subsample'],
    },
    'MLPClassifier': {
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'max_iter': [200, 300, 500, 1000],
        'alpha': [1, 0.1, 0.001, 0.0001],
    },

}


class SKLearnModel(BaseModel):

    def __init__(self, classifier, balancer=None):
        self.balancer = balancer
        self.classifier = classifier

    @abc.abstractmethod
    def get_x_train(self, dataset: BaseDataset):
        raise NotImplementedError

    @abc.abstractmethod
    def get_x_test(self, dataset: BaseDataset):
        raise NotImplementedError

    @abc.abstractmethod
    def get_y_vector(self, dataset: BaseDataset):
        raise NotImplementedError

    @abc.abstractmethod
    def get_x(self, record):
        raise NotImplementedError

    def train(self, dataset: BaseDataset):
        x_train = self.get_x_train(dataset)
        y_train = self.get_y_vector(dataset)

        if self.balancer:
            x_train, y_train = self.balancer.fit_resample(x_train, y_train)

        self.classifier.fit(x_train, y_train)

    def predict_one(self, record: BaseDataset.Record) -> Any:
        return self.classifier.predict([self.get_x(record)])[0]

    def predict_many(self, dataset: BaseDataset) -> list:
        return self.classifier.predict(self.get_x_test(dataset))

    def tune(self, train_dataset: BaseDataset, param_grid: dict or list or None = None):
        param_grid = param_grid or PARAMS_GRID.get(self.classifier.__class__.__name__)
        if param_grid is None:
            logger.error(
                'Grid parameters is None and there is no default parameters for this classifier. Quitting.'
            )
            return

        grid_search_cv = GridSearchCV(
            estimator=self.classifier,
            param_grid=param_grid,
        )
        grid_search_cv.fit(
            self.get_x_train(train_dataset),
            self.get_y_vector(train_dataset),
        )
        self.classifier = self.classifier.__class__(**grid_search_cv.best_params_)

    def evaluate(self, dataset: BaseDataset, as_dict: bool = False):
        y_test = self.get_y_vector(dataset)
        y_pred = self.predict_many(dataset)

        if as_dict:
            return {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1-score': f1_score(y_test, y_pred, average='macro')
            }
        print()
        print(confusion_matrix(y_test, y_pred))
        print()
        print(classification_report(y_test, y_pred))
        print()
        print(f'F1-score: {f1_score(y_test, y_pred, average="macro")}')
