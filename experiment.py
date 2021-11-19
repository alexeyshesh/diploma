import re
import pandas as pd
import nltk
from tqdm import tqdm
from src import deptree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def tokenize(text):
    sents = nltk.sent_tokenize(text)
    v = []
    for s in sents:
        v.append(deptree.DepTree(s).tree_str)

    return v


vectorizer = TfidfVectorizer(analyzer='word', tokenizer=tokenize, max_features=1500, min_df=5, max_df=0.7)

train_df = pd.read_csv('/Users/alexeyshesh/author_attribution/datasets/russian_classics/train_4_10_1000.csv')
print(train_df.shape)

test_df = pd.read_csv('/Users/alexeyshesh/author_attribution/datasets/russian_classics/test_4_10_600.csv')
print(test_df.shape)

train_texts = train_df['Content']
test_texts = test_df['Content']

train_X, train_y = list(train_texts), [int(a) for a in train_df['Author']]
test_X, test_y = list(test_texts), [int(a) for a in test_df['Author']]

all_texts = list(train_texts) + list(test_texts)
y = train_y + test_y

tfidfconverter = vectorizer
X = tfidfconverter.fit_transform(all_texts).toarray()

X_train, X_test, y_train, y_test = X[:len(train_y)], X[len(train_y):], train_y, test_y

classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
