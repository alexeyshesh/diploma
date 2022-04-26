import re
import pandas as pd
import pymorphy2
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from tqdm import trange, tqdm

morph = pymorphy2.MorphAnalyzer()

with open('../data/functional_words/vvodn') as f:
    VVODN_LIST = f.read().split()  # список вводных слов русского языка

with open('../data/functional_words/souz') as f:
    SOUZ_LIST = f.read().split() # список союзов русского языка

with open('../data/functional_words/chast') as f:
    CHAST_LIST = f.read().split()  #  список частиц русского языка

with open('../data/functional_words/judge') as f:
    JUDGE_LIST = f.read().split()  # список оценочных слов русского языка


def syntax_encode(text):
    text = text.lower()
    text = re.findall('\w+|\.\.\.|\!\?|[:\-,\.;\(\)\!\?]', text)
    result = []
    for token in text:
        pos = str(morph.parse(token)[0].tag.POS or morph.parse(token)[0].tag)
        if pos == 'PNCT':
            result.append(token)
            continue 
        result.append(pos)
    return result

def tokenize(text):
    res = []
    for sent in nltk.sent_tokenize(text):
        t = syntax_encode(sent)
        s = nltk.ngrams(t, 3)
        res += s
    return res

def vectorize(text):
    text = re.findall('[а-яё]+', text.lower())
    vvodn = dict.fromkeys(VVODN_LIST, 0)
    souz = dict.fromkeys(SOUZ_LIST, 0)
    chast = dict.fromkeys(CHAST_LIST, 0)
    judge = dict.fromkeys(JUDGE_LIST, 0)
    for word in text:
        if word in vvodn:
            vvodn[word] += 1
        if word in souz:
            souz[word] += 1
        if word in chast:
            chast[word] += 1
        if morph.parse(word)[0].normal_form in judge:
            judge[morph.parse(word)[0].normal_form] += 1

    res = []
    for key in sorted(vvodn):
        res.append(vvodn[key] / max(1, sum(vvodn.values())))
    for key in sorted(souz):
        res.append(souz[key] / max(1, sum(souz.values())))
    for key in sorted(chast):
        res.append(chast[key] / max(1, sum(chast.values())))
    for key in sorted(judge):
        res.append(judge[key] / max(1, sum(judge.values())))

    return res

def predict(df, test_text, thread=None):
    vectorizer = TfidfVectorizer(analyzer='word', tokenizer=tokenize, max_features=1000, min_df=5, max_df=0.7)
    all_texts = list(df['Content']) + [test_text]
    y = list(df['Author'])
    thread._signal.emit("Обработка синтаксиса текста")
    tfidfconverter = vectorizer
    X = tfidfconverter.fit_transform(all_texts).toarray()
    X1 = []
    
    thread._signal.emit("Обработка слов закрытых грамматических классов")
    for i in range(len(y) + 1):
        thread._signal.emit(str(int(i / (len(y) + 1) * 100)))
        X1.append(list(X[i]) + vectorize(all_texts[i]))

    X = X1
    thread._signal.emit("Построение дерева решений")
    classifier = RandomForestClassifier(n_estimators=2000, random_state=0)
    classifier.fit(X[:-1], y)

    pred = classifier.predict(X[-1:])
    print(classifier.predict_proba(X[-1:]))
    return pred[0]