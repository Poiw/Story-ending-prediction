import torch
import torch.nn as nn
import paramcfg

options = paramcfg.options

class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        
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
            nn.Conv1d(200, 400, kernel_size=2, stride=2),
            nn.LeakyReLU()
        )

        self.conv31 = nn.Sequential(
                    nn.Conv1d(400, 400, kernel_size=2),
                    nn.LeakyReLU()
        )

        self.conv32 = nn.Sequential(
            nn.Conv1d(400, 800, kernel_size=2, stride=2),
            nn.LeakyReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(800, 800, kernel_size=2, stride=1),
            nn.LeakyReLU()
        )

    def forward(self, x):

        x1 = self.conv11(x)
        y1 = self.conv12(x1)

        print(y1.shape)

        x2 = self.conv21(y1)
        y2 = self.conv22(x2)

        print(y2.shape)

        x3 = self.conv31(y2)
        y3 = self.conv32(x3)

        print(y3.shape)

        out = self.conv4(y3)

        print(out.shape)

        return out

class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(800, 100, 5),
            nn.LeakyReLU()
        )

        self.sig = nn.Sigmoid()

    def forward(self, x1, x2, x3, x4, y):

        x = torch.cat((x1, x2, x3, x4, y), dim=2)

        xx = self.conv1(x)

        maxx = torch.max(xx, dim=1)[0]

        out = maxx.reshape(maxx.shape[0])

        return out