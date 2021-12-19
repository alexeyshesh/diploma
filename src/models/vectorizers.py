from src.utils import get_by_ngrams

from sklearn.feature_extraction.text import TfidfVectorizer


class TrigramTfIdfVectorizer:
    def __new__(cls, *args, **kwargs):
        return TfidfVectorizer(tokenizer=get_trigrams)


class BigramTfIdfVectorizer:
    def __new__(cls, *args, **kwargs):
        return TfidfVectorizer(tokenizer=get_bigrams)


class WordTrigramTfIdfVectorizer:
    def __new__(cls, *args, **kwargs):
        return TfidfVectorizer(tokenizer=get_trigrams)


def get_trigrams(preprocessed_text):
    trigrams = []
    for word in preprocessed_text.split():
        if len(word) >= 3:
            trigrams += get_by_ngrams(word, 3)

    return trigrams


def get_bigrams(preprocessed_text):
    trigrams = []
    for word in preprocessed_text.split():
        if len(word) >= 2:
            trigrams += get_by_ngrams(word, 2)

    return trigrams


def get_word_trigrams(preprocessed_text):
    return get_by_ngrams(preprocessed_text.split(), 3)
