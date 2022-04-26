import os
import pandas as pd
from collections import Counter
from tqdm import tqdm, trange
import re
from nltk import sent_tokenize


def count_words(text):
    """Считает количество слов в предложении"""
    return len(re.findall("\w+", text))


def sample_split(text, sample_size=1000):
    """Создает из текста список его частей длиной чуть меньше sample_size"""
    samples = []
    cur_sample = ''
    for sent in sent_tokenize(text):
        cur_sample += ' ' + sent 
        if count_words(sent) + count_words(cur_sample) > sample_size:
            samples.append(cur_sample)
            cur_sample = '' 
    if cur_sample != '':
        samples.append(cur_sample)
    return samples 


def create_dataset(path: str, sample_len=1000, app=None, thread=None):
    author_files = get_files(path)
    res = []
    authors = sorted(list(author_files.keys()))
    for author_index in range(len(authors)):
        if thread:
            thread._signal.emit(str(int(author_index / len(authors) * 100)))
            thread._signal.emit("Анализ текстов: " + authors[author_index])
        author = authors[author_index]
        for text_path in author_files[author]:
            text = open(text_path).read()
            sampled_text = sample_split(text, sample_len)
            if len(sampled_text) > 1:
                if count_words(sampled_text[-1]) < sample_len // 2:
                    sampled_text[-2] += sampled_text[-1]
                    sampled_text.pop(-1)
            for sample in sampled_text:
                res.append([author, author_index, sample])
    if thread:
        thread._signal.emit("100")

    df = pd.DataFrame(res, columns=['Author Name', 'Author', 'Content'])
    authors = dict(enumerate(authors))
    return df, authors


def get_files(path: str) -> dict:
    authors = os.listdir(path)
    texts = dict.fromkeys(authors, [])
    for author in authors:
        author_path = os.path.join(path, author)
        if os.path.isdir(author_path):
            author_files = os.listdir(author_path)
            for p in author_files:
                p = os.path.join(author_path, p)
                if os.path.isfile(p):
                    texts[author].append(p)
        else:
            del texts[author]
    return texts
