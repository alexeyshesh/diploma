from sklearn.metrics import classification_report
from tqdm import tqdm

from src.model import AAModel
from src.ds import Dataset


class Attributor:

    def __init__(self, model: AAModel):
        self.model = model

    @staticmethod
    def get_confusion_matrix(dataset: Dataset):
        matrix = {}
        for a1 in dataset.authors:
            matrix[a1] = {}
            for a2 in dataset.authors:
                matrix[a1][a2] = 0

        return matrix

    @staticmethod
    def print_confusion_matrix(matrix: dict):
        authors = sorted(matrix.keys())
        first_col_padding = max([len(a) for a in authors])
        print(' \u001b[1m' * (first_col_padding + 1) + ' '.join(authors), end='\u001b[0m\n\n')
        for correct in authors:
            print(
                '\u001b[1m{name: >{padding}}\u001b[0m'.format(
                    name=correct,
                    padding=first_col_padding,
                ),
                end=' ',
            )
            for predicted in authors:
                template = '{value: >{padding}}\u001b[0m'
                if matrix[correct][predicted] == max(matrix[correct].values()):
                    if correct == predicted:
                        template = '\u001b[1m\u001b[32m' + template
                    else:
                        template = '\u001b[1m\u001b[31m' + template
                print(
                    template.format(
                        value=matrix[correct][predicted],
                        padding=len(predicted),
                    ),
                    end=' ',
                )
            print('')

    def evaluate(self, test_dataset: Dataset, as_dict: bool = False) -> dict or None:
        correct, wrong = 0, 0
        correct_answers, predictions = [], []
        confusion_matrix = self.get_confusion_matrix(test_dataset)
        tqdm_dataset = tqdm(test_dataset)
        for record in tqdm_dataset:
            prediction = self.model.predict(record.text)
            if prediction == record.author:
                correct += 1
            else:
                wrong += 1

            correct_answers.append(record.author)
            predictions.append(prediction)
            confusion_matrix[record.author][prediction] += 1

            tqdm_dataset.set_description(f'Accuracy: {(correct / (correct + wrong)):3.2f}')

        if as_dict:
            return {
                'correct': correct,
                'wrong': wrong,
                'accuracy': correct / (correct + wrong),
                'confusion_matrix': confusion_matrix,
            }

        print(f'\n\u001b[1mCorrect / Wrong:\u001b[0m {correct} / {wrong}')
        print(f'\u001b[1mAccuracy:\u001b[0m {correct / (correct + wrong)}\n\n')
        self.print_confusion_matrix(confusion_matrix)
        print('\n\n')
        report = classification_report(y_true=correct_answers, y_pred=predictions)
        print(report)
