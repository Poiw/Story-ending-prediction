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

logging.basicConfig(filename='../log/train.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

options = paramcfg.options

def validation(ValLoader, fnet, predictor):

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
                body.append(fnet(s))
            for s in y:
                ends.append(fnet(s))

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

    trainwriter = SummaryWriter('../log/train')
    valwriter = SummaryWriter('../log/validation')

    model = gensim.models.Word2Vec.load(options.embeddingmodel)
    model_dict = model.wv

    # load train data
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
    # length = int(len(csvdata)/10*9)
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
    for i in range(len(csvdata)):
        story = list(csvdata.loc[i])
        valdata.append(story[:-1])
        vallabels.append(int(story[-1]))
    valSet = data.ValidDataSet(valdata, model_dict, vallabels)
    ValLoader = DataLoader(valSet, batch_size=1, shuffle=False, num_workers=4, drop_last=False, collate_fn=data.valid_collate_fn)

    # fnet = netModel.FeatureNet()
    fnet = netModel.BiLSTM()
    fnet.cuda()
    predictor = netModel.Predictor(200)
    predictor.cuda()

    opt_f = torch.optim.Adam(fnet.parameters(), lr=options.LR)
    opt_p = torch.optim.Adam(predictor.parameters(), lr=options.LR)
    lossfunc = torch.nn.MSELoss()


    for epoch in range(100000):

        logging.info('epoch: {}'.format(epoch))

        Loss = 0
        for step, (sentences, labels) in enumerate(TrainLoader):

            features = []

            sentences = sentences.cuda()
            labels = labels.cuda()

            for s in sentences:
                features.append(fnet(s))

            output = predictor(features[0], features[1], features[2], features[3], features[4])

            loss = lossfunc(output, labels)

            opt_f.zero_grad()
            opt_p.zero_grad()
            
            loss.backward()

            opt_f.step()
            opt_p.step()

            # print(' [{}/{}]: {}'.format(step, len(TrainLoader), loss.item()))
            Loss = (Loss * step + loss.item()) / (step + 1)

        logging.info('Epoch Loss: {}'.format(Loss))
        trainwriter.add_scalar('loss', Loss, epoch)

        acc = validation(ValLoader, fnet, predictor)

        logging.info('validation acc: {}'.format(acc))
        valwriter.add_scalar('acc', acc, epoch)




if __name__ == '__main__':
    print('start.')
    main()
    print('done.')