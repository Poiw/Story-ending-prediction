import torch
import pandas
import gensim
import logging
import os
import numpy as np 
import paramcfg
import data
import netModel

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

options = paramcfg.options

def main():

    model = gensim.models.Word2Vec.load(options.embeddingmodel)
    model_dict = model.wv

    # load train data
    csvdata = pandas.read_csv('../Data/train.csv')
    traindata = []
    for i in range(len(csvdata)):
        story = list(csvdata.loc[i])
        traindata.append(story)

    trainSet = data.TrainDataSet(traindata, model_dict)
    TrainLoader = DataLoader(trainSet, batch_size=options.BS, shuffle=True, num_workers=4, drop_last=False, collate_fn=data.train_collate_fn)


    csvdata = pandas.read_csv('../Data/val.csv')
    valdata = []
    vallabels = []
    for i in range(len(csvdata)):
        story = list(csvdata.loc[i])
        valdata.append(story[:-1])
        vallabels.append(int(story[-1]))
    valSet = data.ValidDataSet(valdata, model_dict)
    ValLoader = DataLoader(valSet, batch_size=1, shuffle=False, num_workers=4, drop_last=False, collate_fn=data.valid_collate_fn)

    