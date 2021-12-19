from nltk import sent_tokenize

from src.deptree import get_sentence_hash
from src.models.vector_model import SKVectorModel


class StructureModel(SKVectorModel):

    def __init__(self, classifier, vectorizer):
        super().__init__(classifier, vectorizer, structurize_text)


def structurize_text(text):
    result = []
    for sent in sent_tokenize(text):
        result.append(get_sentence_hash(sent))

    return '^' + '$ ^'.join(result) + '$'
