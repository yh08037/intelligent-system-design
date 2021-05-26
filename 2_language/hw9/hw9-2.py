# ID: 2018115809 (undergraduate)
# NAME: Dohun Kim
# File name: hw9-2.py
# Platform: Python 3.7.4 on Ubuntu Linux 18.04
# Required Package(s): os, sys, numpy=1.19.2, matplotlib=3.3.4, scikit-learn=0.24.1

'''
hw9-2.py :
    Ch03. Word2Vec, CBOW, Skipgram
'''

############################## import required packages ##############################
# coding: utf-8
import sys, os
sys.path.append(os.pardir)

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from common.layers import MatMul
from common.util import preprocess, create_contexts_target, convert_one_hot
from common.trainer import Trainer
from common.optimizer import Adam
from simple_cbow import SimpleCBOW
from simple_skip_gram import SimpleSkipGram


#################################### CBOW predict ####################################
print('CBOW predict')

# 샘플 맥락 데이터
c0 = np.array([[1, 0, 0, 0, 0, 0, 0]])
c1 = np.array([[0, 0, 1, 0, 0, 0, 0]])

# 가중치 초기화
W_in = np.random.randn(7, 3)
W_out = np.random.randn(3, 7)

# 계층 생성
in_layer0 = MatMul(W_in)
in_layer1 = MatMul(W_in)
out_layer = MatMul(W_out)

# 순전파
h0 = in_layer0.forward(c0)
h1 = in_layer1.forward(c1)
h = 0.5 * (h0 + h1)  # average
s = out_layer.forward(h)  # score

print(s)
print('-' * 50)


################################# Context and Target ################################
print('Context and Target')

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

print('corpus\n ', corpus)
print('id_to_word\n ', id_to_word)

contexts, target = create_contexts_target(corpus, window_size=1)

print('context')
print(contexts)
print('target\n ', target)

print('-' * 50)


############################# Convert to One-hot Vector ############################
print('Convert to One-hot Vector')

vocab_size = len(word_to_id)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

print('context')
print(contexts)
print('target')
print(target)

print('-' * 50)

################################# Train SimpleCBOW #################################
print('Train SimpleCBOW')

window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

# initialize model
model = SimpleCBOW(vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

# train model
trainer.fit(contexts, target, max_epoch, batch_size)

# plot learning curve
trainer.plot()

# Word Embedding - W_in
word_vecs1 = model.word_vecs1
for word_id, word in id_to_word.items():
    print(word, word_vecs1[word_id])

# Word Embedding - W_out
word_vecs2 = model.word_vecs2
for word_id, word in id_to_word.items():
    print(word, word_vecs2[word_id])

print('-' * 50)


#################################### t-SNE plot ####################################

# 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False

tsne = TSNE(n_components=2)

# 100개의 단어에 대해서만 시각화
X_tsne = tsne.fit_transform(word_vecs2)

vocab = list(id_to_word.values())

df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])

fig = plt.figure()
fig.set_size_inches(20, 10)
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    ax.annotate(word, pos)
plt.show()


############################## Train SimpleSkipGram ##############################
print('Train SimpleSkipGram')

window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

# initialize model
model = SimpleSkipGram(vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

# train model
trainer.fit(contexts, target, max_epoch, batch_size)

# plot learning curve
trainer.plot()

# Word Embedding - W_in
word_vecs1 = model.word_vecs1
for word_id, word in id_to_word.items():
    print(word, word_vecs1[word_id])

# Word Embedding - W_out
word_vecs2 = model.word_vecs2
for word_id, word in id_to_word.items():
    print(word, word_vecs2[word_id])

print('-' * 50)


#################################### t-SNE plot ####################################

# 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False

tsne = TSNE(n_components=2)

# 100개의 단어에 대해서만 시각화
X_tsne = tsne.fit_transform(word_vecs2)

vocab = list(id_to_word.values())

df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])

fig = plt.figure()
fig.set_size_inches(20, 10)
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    ax.annotate(word, pos)
plt.show()