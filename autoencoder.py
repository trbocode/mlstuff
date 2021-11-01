import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.module):
    def __init__(self,params):
        super(Net, self).__init__()

        # Encoding
        self.conv1=nn.Conv2d(params['nc'],params['nf'],4,2,1)
        self.conv2=nn.Conv2d(params['nf'],2*params['nf'],4,2,1)
        self.conv3=nn.Conv2d(2*params['nf'],4*params['nf'],4,2,1)
        self.conv4=nn.Conv2d(4*params['nf'],params['ncode'],4,1,0)

        # Decoding
        self.anticonv1=nn.ConvTranspose2d(params['ncode'],params['nf']*4,4,1,0)
        self.anticonv2=nn.ConvTranspose2d(params['nf']*4,params['nf']*2,4,2,1)
        self.anticonv3=nn.ConvTranspose2d(params['nf']*2,params['nf'],4,2,1)
        self.anticonv4=nn.ConvTranspose2d(params['nf'],params['nc'],4,2,1)
        
    def forward(self,x):
        x=F.BatchNorm2d(3)
        x=F.LeakyReLU(self.conv1(x))
        x=F.LeakyReLU(self.conv2(x))
        x=F.LeakyReLU(self.conv3(x))
        x=F.LeakyReLU(self.conv4(x))
        x=F.LeakyReLU(self.anticonv1(x))
        x=F.LeakyReLU(self.anticonv2(x))
        x=F.LeakyReLU(self.anticonv3(x))
        x=F.LeakyReLU(self.anticonv4(x))

criterion=nn.MSELoss()
optimizer=optim.Adam(Net.parameters(),lr=0.001)
