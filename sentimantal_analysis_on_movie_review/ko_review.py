# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 22:04:23 2018

@author: LAB_SAN
"""


def wtf(a):
    print(type(a))
    try:
        print(a.shape)
        print(a.info())
    except:
        print(len(a))


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re
import konlpy
from konlpy.tag import Twitter
import matplotlib.pyplot as plt
% matplotlib
inline
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from numpy import nan
from bs4 import BeautifulSoup
from math import sqrt
from matplotlib import rc
import os

rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

os.chdir('C:\\Users\\SAN\\Google ����̺�\\1�Ѿ���п�\\���ΰ���\\2ĳ�� ���͵�\\Sentiment Analysis on Movie Reviews/')
train_data = pd.read_csv('ratings_train.txt', encoding='utf-8', delimiter='\t', keep_default_na=False)
test_data = pd.read_csv('ratings_test.txt', encoding='utf-8', delimiter='\t', keep_default_na=False)

test_data.info()
train_data.info()

Sentiment_words = []
for row in train_data['label']:
    if row == 0:
        Sentiment_words.append('negative')
    elif row == 1:
        Sentiment_words.append('positive')
    else:
        Sentiment_words.append('Failed')

train_data['Sentiment_words'] = Sentiment_words

# ī��Ʈ �� �ð�ȭ
word_count = pd.value_counts(train_data['Sentiment_words'].values, sort=False)
word_count

Index = [1, 2]
plt.figure(figsize=(15, 5))
plt.bar(Index, word_count, color='blue')
plt.xticks(Index, ['negative', 'positive'], rotation=45)
plt.ylabel('word_count')
plt.xlabel('word')
plt.title('Count of Moods')
plt.bar(Index, word_count)
for a, b in zip(Index, word_count):
    plt.text(a, b, str(b), color='red', fontweight='bold')


# ������ ����
def review_to_words(raw_review):
    review = raw_review
    review = re.sub('[^��-�R]', ' ', review)
    review = review.split()
    return (' '.join(review))


wtf(train_data)
wtf(test_data)

corpus = []
for i in range(0, 150000):
    corpus.append(review_to_words(train_data['document'][i]))

corpus1 = []
for i in range(0, 50000):
    corpus1.append(review_to_words(test_data['document'][i]))

# ������ ���� �����Ȱ� ����
train_data['new_document'] = corpus
train_data.drop(['document'], axis=1, inplace=True)

train_data.head()

# new_phrase �� word �� ��ġ�°�.  // �ڿ� split_word �� ���� �� �𸣰ٴ�.
words = ' '.join(train_data['new_document'])
split_word = " ".join([word for word in words.split()])

# ���� Ŭ����
wordcloud = WordCloud(relative_scaling=0.2,
                      background_color='black',
                      width=3000,
                      height=2500
                      ).generate(split_word)

plt.figure(1, figsize=(13, 13))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

pos = train_data['new_document']
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer="word",
                             preprocessor=None,
                             stop_words=None,
                             max_features=2000)

pos_words = vectorizer.fit_transform(pos)
pos_words = pos_words.toarray()

pos = vectorizer.get_feature_names()

wtf(pos)  # pos �� �����ϴ� ����
wtf(pos_words)  # �̰� ī��Ʈ���� (�������ڵ�?)

dist = np.sum(pos_words, axis=0)
for tag, count in zip(pos, dist):
    print (tag, count)

wtf(dist)
postive_new = pd.DataFrame(dist)
postive_new.columns = ['word_count']

postive_new['word'] = pd.Series(pos, index=postive_new.index)
postive_new1 = postive_new[['word', 'word_count']]

wtf(postive_new1)

top_30_words = postive_new1.sort_values(['word_count'], ascending=[0])

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=1500)
x__train = cv.fit_transform(corpus).toarray()
x__test = cv.fit_transform(corpus1).toarray()
y = train_data.iloc[:, 1].values

wtf(train_data)
train_data.iloc[:, 1].values

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x__train, y, test_size=0.40, random_state=0)

from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_real_pred = classifier.predict(x__test)

mse = ((y_pred - y_test) ** 2).mean()
mse

rmse = sqrt(mse)
rmse

wtf(train_data)
type(x__train)
wtf(x__train)
train_data['label']

text_train, y_train = x__train, train_data['label'].as_matrix()












