import torch
import torch.nn as nn
import paramcfg
import functions

options = paramcfg.options

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv11 = nn.Sequential(
            nn.Conv1d(options.dim, 100, kernel_size=2, padding=1),
            nn.LeakyReLU()
        )

        self.conv12 = nn.Sequential(
            nn.Conv1d(100, 200, kernel_size=2, stride=2),
            nn.LeakyReLU()
        )

        self.conv21 = nn.Sequential(
                    nn.Conv1d(200, 200, kernel_size=2, padding=1),
                    nn.LeakyReLU()
        )

        self.conv22 = nn.Sequential(
            nn.Conv1d(200, 200, kernel_size=2, stride=2),
            nn.LeakyReLU()
        )

        self.conv31 = nn.Sequential(
            nn.Conv1d(200, 200, kernel_size=2),
            nn.LeakyReLU()
        )

        self.conv32 = nn.Sequential(
            nn.Conv1d(200, 200, kernel_size=2, stride=2),
            nn.LeakyReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(200, 200, kernel_size=2, stride=1),
            nn.LeakyReLU()
        )

    def forward(self, x):

        x1 = self.conv11(x)
        y1 = self.conv12(x1)

        # print(y1.shape)

        x2 = self.conv21(y1)
        y2 = self.conv22(x2)

        # print(y2.shape)

        x3 = self.conv31(y2)
        y3 = self.conv32(x3)

        # print(y3.shape)

        out = self.conv4(y3)

        out = torch.max(out, dim=2, keepdim=True)[0]

        # print(out.shape)

        return out

class BiLSTM(nn.Module): 
    def __init__(self):
        super(BiLSTM, self).__init__()

        self.bilstm = nn.LSTM(input_size=100, hidden_size=100, num_layers=options.LSTMLayer, bidirectional=True)

    def forward(self, x):

        x = x.permute([2, 0, 1])

        h0 = torch.zeros((options.LSTMLayer*2, x.shape[1], x.shape[2])).cuda()
        c0 = torch.zeros((options.LSTMLayer*2, x.shape[1], x.shape[2])).cuda()

        output, (hn, cn) = self.bilstm(x, (h0, c0))

        output = torch.max(output, dim=0)[0]
        output = output.reshape(output.shape[0], output.shape[1], 1)

        return output

class Predictor(nn.Module):
    def __init__(self, channel=800):
        super(Predictor, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(channel, 100, 5),
            nn.LeakyReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(100, 2, 1),
            nn.LeakyReLU()
        )

        self.softmax = nn.Softmax(dim=1)
        self.sig = nn.Sigmoid()

    def forward(self, x1, x2, x3, x4, y):

        x = torch.cat((x1, x2, x3, x4, y), dim=2)
        xx = self.conv1(x)
        # xxx = self.conv2(xx)
        maxx = torch.max(xx, dim=1)[0]

        out = self.sig(maxx)
        out = out.reshape(out.shape[0])

        # out = self.softmax(xxx)
        # out = out[:,0,0]
        
        return out

class BiLSTMModule(nn.Module):
    def __init__(self):
        super(BiLSTMModule, self).__init__()

        self.bilstm = nn.LSTM(input_size=100, hidden_size=100, num_layers=options.LSTMLayer, bidirectional=True)

    def forward(self, x):

        x = x.permute([2, 0, 1])

        h0 = torch.zeros((options.LSTMLayer*2, x.shape[1], x.shape[2])).cuda()
        c0 = torch.zeros((options.LSTMLayer*2, x.shape[1], x.shape[2])).cuda()

        output, (hn, cn) = self.bilstm(x, (h0, c0))

        return output


class Predictor_FC(nn.Module):
    def __init__(self):
        super(Predictor_FC, self).__init__()

        self.bodyfc = nn.Sequential(
            nn.Linear(200, 100),
            nn.SELU()
        )

        self.endfc = nn.Sequential(
            nn.Linear(200, 100),
            nn.SELU()
        )

        self.fc = nn.Sequential( 
            nn.Linear(400, 1),
            nn.Sigmoid()
        )

    def forward(self, body_raw, end_raw):
        # body length * batchsize * hiddensize
        # end length * batchsize * hiddensize
        # label batchsize

        body = self.bodyfc(body_raw)
        end = self.endfc(end_raw)
        
        # batchsize * hiddensize * 1
        AoA_h_end = functions.AoA(body, end)

        # batchsize * hiddensize * 1
        maxh_end = (torch.max(end, dim=0, keepdim=True)[0]).permute([1, 2, 0])

        # batchsize * hiddensize * 1
        AoA_h_body = functions.AoA(end, body)

        # batchsize * hiddensize * 1
        maxh_body = (torch.max(body, dim=0, keepdim=True)[0]).permute([1, 2, 0])

        feature = torch.cat((AoA_h_end, maxh_end, AoA_h_body, maxh_body), dim=2)

        feature = torch.flatten(feature, start_dim=1)

        output = self.fc(feature).reshape(-1)

        return output

