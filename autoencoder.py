import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=nn.Conv2d(1,6,5)
        self.conv2=nn.Conv2d(6, 1, 10)
        self.anticonv1=nn.ConvTranspose2d(1,6,10)
        self.anticonv2=nn.ConvTranspose2d(6,1,5)
    def forward(self,x):
        x=F.BatchNorm2d(3)
        x=F.LeakyReLU(self.conv1(x))
        x=F.LeakyReLU(self.conv2(x))
        x=F.LeakyReLU(self.anticonv1(x))
        x=F.LeakyReLU(self.anticonv2(x))

criterion=nn.MSELoss()
optimizer=optim.Adam(net.parameters(),lr=0.001)
