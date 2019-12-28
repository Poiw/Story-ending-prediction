import gensim
import logging
import os
import paramcfg
import pandas

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 
options = paramcfg.options

sentences = []
data = pandas.read_csv('../Data/train.csv')
maxlength = 0
for column in data.columns:
    print(column)
    tmp = list(data[column].loc[:])
    for idx in range(len(tmp)):
        tmp[idx] = tmp[idx].split()[:-1]
        maxlength = max(maxlength, len(tmp[idx]))
    sentences = sentences + tmp

data = pandas.read_csv('../Data/val.csv')
for column in data.columns:
    print(column)
    if column == 'AnswerRightEnding':
        continue
    tmp = list(data[column].loc[:])
    for idx in range(len(tmp)):
        tmp[idx] = tmp[idx].split()[:-1]
        maxlength = max(maxlength, len(tmp[idx]))
    sentences = sentences + tmp

data = pandas.read_csv('../Data/test.csv')
for column in data.columns:
    print(column)
    if column == 'AnswerRightEnding':
        continue
    tmp = list(data[column].loc[:])
    for idx in range(len(tmp)):
        tmp[idx] = tmp[idx].split()[:-1]
        maxlength = max(maxlength, len(tmp[idx]))
    sentences = sentences + tmp
 
fname = options.embeddingmodel
if os.path.exists(fname):
    # load the file if it has already been trained, to save repeating the slow training step below
    model = gensim.models.Word2Vec.load(fname)
else:
    # can take a few minutes, grab a cuppa
    model = gensim.models.Word2Vec(sentences, size=100, min_count=1, workers=4, iter=200) 
    model.save(fname)

print('Max Length: ', maxlength)
