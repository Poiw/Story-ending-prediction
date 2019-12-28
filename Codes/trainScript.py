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
import math

logging.basicConfig(filename='../log/train.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

options = paramcfg.options

def focal_loss(output, labels):

    return -torch.mean(torch.log(1 - output + 1e-4) * (1 - labels) + torch.log(output + 1e-4) * labels)

    # loss = 0
    # for out, gth in zip(output, labels):
    #     loss = loss + ( - torch.log(1.0 - out) * ((1.0-gth)**2) - torch.log(out) * (gth**2) )
    
    # return loss / len(output)

def validation(ValLoader, fnetlstm, fnetcnn, predictor):

    success = 0
    failed = 0

    with torch.no_grad():
        logging.info('validating...')
        for x, y, labels in ValLoader:

            x = x.cuda()
            y = y.cuda()
            
            body = []
            ends = []
            for s in x:
                body.append(torch.cat((fnetlstm(s), fnetcnn(s)), dim=1))
            for s in y:
                ends.append(torch.cat((fnetlstm(s), fnetcnn(s)), dim=1))

            batchsize = len(labels)

            score1 = predictor(body[0], body[1], body[2], body[3], ends[0])
            score2 = predictor(body[0], body[1], body[2], body[3], ends[1])
            
            for i in range(batchsize):
                result = 1 if score1[i] > score2[i] else 2

                if result == labels[i]:
                    success += 1
                else:
                    failed += 1
            
    return success / (success + failed)

def main():

    logging.info('{}'.format(options))

    model = gensim.models.Word2Vec.load(options.embeddingmodel)
    model_dict = model.wv

    # load train data
    length = 0
    csvdata = pandas.read_csv('../Data/train.csv')
    traindata = []
    trainlabels = []
    for i in range(len(csvdata)):
        story = list(csvdata.loc[i])
        traindata.append(story)
        trainlabels.append(1.0)

        if torch.rand(()) > 0.5:
            index = torch.randint(4, ()).item()

            tmp = story[index]
            tmps = story[index+1:5]
            story[index:-1] = tmps
            story[4] = tmp

            traindata.append(story)
            trainlabels.append(0.0)

    # csvdata = pandas.read_csv('../Data/val.csv')
    # length = int(len(csvdata)/5*4)
    # traindata = []
    # trainlabels = []
    # for i in range(length):
    #     story = list(csvdata.loc[i])

    #     endlabel = int(story[-1])
    #     if endlabel == 1:
    #         traindata.append(story[:5])
    #         trainlabels.append(1.0)

    #         traindata.append(story[:4] + story[5:6])
    #         trainlabels.append(0.0)     
    #     else:
    #         traindata.append(story[:5])
    #         trainlabels.append(0.0)

    #         traindata.append(story[:4] + story[5:6])
    #         trainlabels.append(1.0)     

    trainSet = data.TrainDataSet(traindata, model_dict, trainlabels)
    TrainLoader = DataLoader(trainSet, batch_size=options.BS, shuffle=True, num_workers=4, drop_last=False, collate_fn=data.train_collate_fn)


    csvdata = pandas.read_csv('../Data/val.csv')
    valdata = []
    vallabels = []
    for i in range(length, len(csvdata)):
        story = list(csvdata.loc[i])
        valdata.append(story[:-1])
        vallabels.append(int(story[-1]))
    valSet = data.ValidDataSet(valdata, model_dict, vallabels)
    ValLoader = DataLoader(valSet, batch_size=options.BS, shuffle=False, num_workers=4, drop_last=False, collate_fn=data.valid_collate_fn)

    # fnet = netModel.FeatureNet()
    fnetlstm = netModel.BiLSTM()
    fnetlstm.cuda()
    fnetcnn = netModel.CNN()
    fnetcnn.cuda()
    predictor = netModel.Predictor(400)
    predictor.cuda()

    opt_lstm = torch.optim.Adam(fnetlstm.parameters(), lr=options.LR)
    opt_cnn = torch.optim.Adam(fnetcnn.parameters(), lr=options.LR)
    opt_p = torch.optim.Adam(predictor.parameters(), lr=options.LR)
    lossfunc = focal_loss


    for epoch in range(100000):

        logging.info('epoch: {}'.format(epoch))

        Loss = 0
        for step, (sentences, labels) in enumerate(TrainLoader):

            features = []

            sentences = sentences.cuda()
            labels = labels.cuda()

            for s in sentences:
                features.append(torch.cat((fnetlstm(s), fnetcnn(s)), dim=1))

            output = predictor(features[0], features[1], features[2], features[3], features[4])

            loss = lossfunc(output, labels)

            opt_lstm.zero_grad()
            opt_cnn.zero_grad()
            opt_p.zero_grad()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fnetlstm.parameters(), max_norm=0.2, norm_type=2)
            torch.nn.utils.clip_grad_norm_(fnetcnn.parameters(), max_norm=0.2, norm_type=2)
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=0.5, norm_type=2)

            opt_lstm.step()
            opt_cnn.step()
            opt_p.step()

            # print(' [{}/{}]: {}'.format(step, len(TrainLoader), loss.item()))
            Loss = (Loss * step + loss.item()) / (step + 1)

        logging.info('Epoch Loss: {}'.format(Loss))

        acc = validation(ValLoader, fnetlstm, fnetcnn, predictor)

        logging.info('validation acc: {}'.format(acc))

        # torch.save(fnet_lstm.state_dict(), '../Models/fnets/epoch' + str(epoch) + '.para')
        # torch.save(fnet_cnn.state_dict(), '../Models/fnets/epoch' + str(epoch) + '.para')
        # torch.save(predictor.state_dict(), '../Models/predictors/epoch' + str(epoch) + '.para')
        logging.info('saving {} epoch models'.format(epoch))



if __name__ == '__main__':
    print('start.')
    main()
    print('done.')