import torch
import pandas
import gensim
import logging
import os
import numpy as np 
import paramcfg
import data
import netModel
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import math
import shutil

options = paramcfg.options

def test(TestLoader, fnetlstm_b, fnetlstm_e, predictor):


    with open('../Results/prediction_wo_val.txt', 'w') as f:

        with torch.no_grad():
            logging.info('validating...')
            for body, end1, end2, labels in TestLoader:

                body = body.cuda()
                end1 = end1.cuda()
                end2 = end2.cuda()
                labels = labels.cuda()

                body_h = fnetlstm_b(body)
                end_h1 = fnetlstm_e(end1)
                end_h2 = fnetlstm_e(end2)

                score1 = predictor(body_h, end_h1)
                score2 = predictor(body_h, end_h2)
                batchsize = len(labels)

                for i in range(batchsize):
                    result = 1 if score1[i] > score2[i] else 2

                    f.write('{}\n'.format(result))

def validation(ValLoader, fnetlstm_b, fnetlstm_e, predictor):

    success = 0
    failed = 0

    with torch.no_grad():
        logging.info('validating...')
        for body, end1, end2, labels in ValLoader:

            body = body.cuda()
            end1 = end1.cuda()
            end2 = end2.cuda()
            labels = labels.cuda()

            body_h = fnetlstm_b(body)
            end_h1 = fnetlstm_e(end1)
            end_h2 = fnetlstm_e(end2)

            score1 = predictor(body_h, end_h1)
            score2 = predictor(body_h, end_h2)
            batchsize = len(labels)

            for i in range(batchsize):
                result = 1 if score1[i] > score2[i] else 2

                if result == labels[i]:
                    success += 1
                else:
                    failed += 1
            
    return success / (success + failed)

def main():

    loading_path = '../Models/AoA_0.1selfdata/'


    model = gensim.models.Word2Vec.load(options.embeddingmodel)
    model_dict = model.wv

    counter = [0, 0, 0]
    csvdata = pandas.read_csv('../Data/val.csv')
    length = 0
    valdata = []
    vallabels = []
    for i in range(length, len(csvdata)):
        story = list(csvdata.loc[i])

        labels = int(story[-1])

        if length > 0 and counter[labels] > counter[3-labels]:
            s4 = story[4]
            s5 = story[5]

            story[4] = s5
            story[5] = s4

            labels = 3 - labels

        counter[labels] += 1

        valdata.append(story[:-1])
        vallabels.append(labels)
    valSet = data.ValidDataSetBE(valdata, model_dict, vallabels)
    ValLoader = DataLoader(valSet, batch_size=options.BS, shuffle=False, num_workers=4, drop_last=False, collate_fn=data.valid_collate_fn_BE)

    csvdata = pandas.read_csv('../Data/test.csv')
    testdata = []
    testlabels = []
    for i in range(0, len(csvdata)):
        story = list(csvdata.loc[i])

        testdata.append(story)
        testlabels.append(1)
    testSet = data.ValidDataSetBE(testdata, model_dict, testlabels)
    TestLoader = DataLoader(testSet, batch_size=1, shuffle=False, num_workers=4, drop_last=False, collate_fn=data.valid_collate_fn_BE)

    # fnet = netModel.FeatureNet()
    fnetlstm_b = netModel.BiLSTMModule()
    fnetlstm_b.load_state_dict(torch.load(loading_path+'lstm_b/best.para'))
    fnetlstm_b.cuda()
    fnetlstm_e = netModel.BiLSTMModule()
    fnetlstm_e.load_state_dict(torch.load(loading_path+'lstm_e/best.para'))
    fnetlstm_e.cuda()
    predictor = netModel.Predictor_FC()
    predictor.load_state_dict(torch.load(loading_path+'predictor/best.para'))
    predictor.cuda()

    acc = validation(ValLoader, fnetlstm_b, fnetlstm_e, predictor)
    print('Acc: ', acc)

    test(TestLoader, fnetlstm_b, fnetlstm_e, predictor)




if __name__ == '__main__':
    print('start.')
    main()
    print('done.')