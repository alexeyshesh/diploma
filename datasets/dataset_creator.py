import os
import pandas as pd
from collections import Counter
from tqdm import tqdm, trange
import re
from nltk import sent_tokenize


def input_int(prompt: str = '', default: int = None) -> int:
    while True:
        try:
            number = input(prompt)
            if number == '' and default is not None:
                return default
            return int(number)
        except:
            pass


def input_directory_path(prompt: str = '', default: str = None) -> int:
    path = input(prompt)
    
    if path == '' and default is not None:
        return default
    
    while not os.path.isdir(path):
        path = input(prompt)
    return path


def count_words(text):
    """Считает количество слов в предложении"""
    return len(re.findall("\w+", text))


def sample_split(text, sample_size=500):
    """Создает из текста список его частей длиной примерно sample_size"""
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


def get_files(path: str):
    """Загрузка списка текстовых файлов из директории path"""
    if path[-1] != '/':
        path += '/'
    return [path + filename for filename in os.listdir(path) if filename[-3:] == 'txt']


def get_author_files(files: list) -> dict:
    """Разделение файлов по автору, указанному в формате автор_название.txt"""
    res = {}
    for filename in files:
        author = filename.split('/')[-1].split('_')[0]
        if author in res:
            res[author].append(filename)
        else:
            res[author] = [filename]

    return res


def create_dataset(author_files: dict, samples=5, sample_len=600, max_authors=30):
    """Создание датасета, где для каждого из max_authors автора есть samples фрагментов текста объемом sample_len слов"""
    res = []
    authors = sorted(list(author_files.keys()))[:max_authors]
    for author_index in trange(len(authors)):
        author = authors[author_index]
        collected_texts = 0
        loop_round = 0
        finished_files = []
        while collected_texts < samples:
            for filename in author_files[author]:
                with open(filename) as f:
                    sample_texts = sample_split(f.read(), sample_len)
                    if loop_round >= len(sample_texts):
                        if filename not in finished_files:
                            finished_files.append(filename)
                        continue 
                    text = sample_texts[loop_round]
                    if len(text) > 0:
                        res.append([authors[author_index], author_index, text])
                        collected_texts += 1
                    if collected_texts >= samples:
                        break
            loop_round += 1
            if len(finished_files) == len(author_files[author]):
                break
        if author_index >= max_authors:
            break

    df = pd.DataFrame(res, columns=['Author Name', 'Author', 'Content'])
    return df


if __name__ == '__main__':
    
    print('=====')
    train_texts_path = input_directory_path('Train text directory: (./data/russian_classics/train) ', default='./data/russian_classics/train')
    test_texts_path = input_directory_path('Test text directory: (./data/russian_classics/test) ', default='./data/russian_classics/test')
    authors_amount = input_int('Authors amount: (all available) ')
    train_texts_amount = input_int('Texts amount in train dataset: (10)', default=10)
    test_texts_amount = input_int('Texts amount in test dataset: (10)', default=10)
    train_words_count = input_int('Words count in train dataset texts: (1000)', default=1000)
    test_words_count = input_int('Words count in test dataset texts: (600)', default=600)
    path_to_save = input_directory_path('Path to save: (./datasets/russian_classics)', default='./datasets/russian_classics')
    
    train_files = get_files(train_texts_path)
    test_files = get_files(test_texts_path)
    
    author_files_train = get_author_files(train_files)
    author_files_test = get_author_files(test_files)
    
    df_train = create_dataset(author_files_train, train_texts_amount, train_words_count, authors_amount)
    df_test = create_dataset(author_files_test, test_texts_amount, test_words_count, authors_amount)
    
    df_train.to_csv('{}/train_{}_{}_{}.csv'.format(path_to_save, authors_amount, train_texts_amount, train_words_count))
    df_test.to_csv('{}/test_{}_{}_{}.csv'.format(path_to_save, authors_amount, test_texts_amount, test_words_count))
    
    print('Done!')
