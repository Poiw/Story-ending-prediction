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


def focal_loss(times = 1):
    def f(output, labels):
        nonlocal times
        return -torch.mean(torch.log(1 - output + 1e-4) * (1 - labels)**times + torch.log(output + 1e-4) * labels**times)
    return f

    # loss = 0
    # for out, gth in zip(output, labels):
    #     loss = loss + ( - torch.log(1.0 - out) * ((1.0-gth)**2) - torch.log(out) * (gth**2) )
    
    # return loss / len(output)

def hinge_loss(output, labels):
    labels[labels < 0.5] = -1
    output = output * 2 - 1
    return torch.mean(torch.max(torch.zeros_like(labels), 1 - labels*output))

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

    if os.path.exists(options.logdir):
        a = input('Are you sure delete ' + options.logdir + '(y/n)')
        if a != 'n':
            shutil.rmtree(options.logdir)
        else:
            exit(0)


    os.mkdir(options.logdir)
    os.mkdir(options.logdir+'lstm_b')
    os.mkdir(options.logdir+'lstm_e')
    os.mkdir(options.logdir+'predictor')

    os.mkdir(options.logdir+'board')
            

    writer = SummaryWriter(options.logdir + 'board/')
    
    logging.basicConfig(filename=options.logdir+'train.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

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

        # if torch.rand(()).item() > 0.9:
        #     if torch.rand(()).item() > 1:
        #         story_random_end = story
        #         index = torch.randint(len(csvdata), ()).item()
        #         other_story = list(csvdata.loc[index])
        #         story_random_end[4] = other_story[4]
        #         traindata.append(story_random_end)
        #         if index == i:
        #             trainlabels.append(1.0)
        #         else:
        #             trainlabels.append(0.0)

        #     else:
        #         index = torch.randint(4, ()).item()

        #         tmp = story[index]
        #         tmps = story[index+1:5]
        #         story[index:-1] = tmps
        #         story[4] = tmp

        #         traindata.append(story)
        #         trainlabels.append(0.0)

    csvdata = pandas.read_csv('../Data/val.csv')
    print(len(csvdata))
    length = int(len(csvdata) * 0.4)
    # traindata = []
    # trainlabels = []
    for i in range(length):
        story = list(csvdata.loc[i])

        endlabel = int(story[-1])
        if endlabel == 1:
            traindata.append(story[:5])
            trainlabels.append(1.0)

            traindata.append(story[:4] + story[5:6])
            trainlabels.append(0.0)     
        else:
            traindata.append(story[:5])
            trainlabels.append(0.0)

            traindata.append(story[:4] + story[5:6])
            trainlabels.append(1.0)     

    counter0 = 0
    counter1 = 0
    for label in trainlabels:
        if label > 0.5:
            counter1 += 1
        else:
            counter0 += 1
    
    logging.info('train data: {}:{}={}'.format(counter0, counter1, counter0/counter1))

    trainSet = data.TrainDataSetBE(traindata, model_dict, trainlabels)
    TrainLoader = DataLoader(trainSet, batch_size=options.BS, shuffle=True, num_workers=4, drop_last=False, collate_fn=data.train_collate_fn_BE)


    counter = [0, 0, 0]
    csvdata = pandas.read_csv('../Data/val.csv')
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
    logging.info('Validation ratio: {}:{}={}'.format(counter[1], counter[2], counter[1]/counter[2]))
    valSet = data.ValidDataSetBE(valdata, model_dict, vallabels)
    ValLoader = DataLoader(valSet, batch_size=options.BS, shuffle=False, num_workers=4, drop_last=False, collate_fn=data.valid_collate_fn_BE)

    # fnet = netModel.FeatureNet()
    fnetlstm_b = netModel.BiLSTMModule()
    fnetlstm_b.cuda()
    fnetlstm_e = netModel.BiLSTMModule()
    fnetlstm_e.cuda()
    predictor = netModel.Predictor_FC()
    predictor.cuda()

    opt_lstm_b = torch.optim.Adam(fnetlstm_b.parameters(), lr=options.LR)
    opt_lstm_e = torch.optim.Adam(fnetlstm_e.parameters(), lr=options.LR)
    opt_p = torch.optim.Adam(predictor.parameters(), lr=options.LR)
    lossfunc = focal_loss(2)

    print('start training...')

    curmax_acc = 0

    with open(options.logdir+'loss.txt','w') as f:


        for epoch in range(100000):

            logging.info('epoch: {}'.format(epoch))

            Loss = 0
            for step, (body, end, labels) in enumerate(TrainLoader):

                body = body.cuda()
                end = end.cuda()
                labels = labels.cuda()

                body_h = fnetlstm_b(body)
                end_h = fnetlstm_e(end)

                output = predictor(body_h, end_h)

                loss = lossfunc(output, labels)

                opt_lstm_b.zero_grad()
                opt_lstm_e.zero_grad()
                opt_p.zero_grad()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fnetlstm_b.parameters(), max_norm=0.2, norm_type=2)
                torch.nn.utils.clip_grad_norm_(fnetlstm_e.parameters(), max_norm=0.2, norm_type=2)
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=0.5, norm_type=2)

                opt_lstm_b.step()
                opt_lstm_e.step()
                opt_p.step()

                if step % 100 == 0:
                    acc = validation(ValLoader, fnetlstm_b, fnetlstm_e, predictor)
                    logging.info('validation acc: {}'.format(acc))

                # print(' [{}/{}]: {}'.format(step, len(TrainLoader), loss.item()))
                Loss = (Loss * step + loss.item()) / (step + 1)

            logging.info('Epoch Loss: {}'.format(Loss))
            writer.add_scalar('train_loss', Loss, epoch)

            acc = validation(ValLoader, fnetlstm_b, fnetlstm_e, predictor)
            writer.add_scalar('val_acc', acc, epoch)

            logging.info('validation acc: {}'.format(acc))
            logging.info('best acc: {}'.format(curmax_acc))

            torch.save(fnetlstm_b.state_dict(), options.logdir+'lstm_b' + '/epoch' + str(epoch) + '.para')
            torch.save(fnetlstm_e.state_dict(), options.logdir+'lstm_e' + '/epoch' + str(epoch) + '.para')
            torch.save(predictor.state_dict(), options.logdir+'predictor' + '/epoch' + str(epoch) + '.para')

            if acc > curmax_acc:
                curmax_acc = acc
                torch.save(fnetlstm_b.state_dict(), options.logdir+'lstm_b' + '/best.para')
                torch.save(fnetlstm_e.state_dict(), options.logdir+'lstm_e' + '/best.para')
                torch.save(predictor.state_dict(), options.logdir+'predictor' + '/best.para')

            logging.info('saving {}th epoch models'.format(epoch))
            f.write('{} {}\n'.format(Loss, acc))



if __name__ == '__main__':
    print('start.')
    main()
    print('done.')