import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self,params):
        super(Net, self).__init__()

        # Encoding
        self.norm=nn.BatchNorm2d(params['nc'])
        self.conv1=nn.Conv2d(params['nc'],params['nf'], 16,4,6)
        self.conv2=nn.Conv2d(params['nf'],2*params['nf'], 16,4,6)
        self.conv3=nn.Conv2d(2*params['nf'],4*params['nf'], 4,2,1)
        self.conv4=nn.Conv2d(4*params['nf'],params['ncode'], 5,1,0)

        # Decoding
        self.anticonv1=nn.ConvTranspose2d(params['ncode'],params['nf']*4,5,1,0)
        self.anticonv2=nn.ConvTranspose2d(params['nf']*4,params['nf']*2,4,2,1)
        self.anticonv3=nn.ConvTranspose2d(params['nf']*2,params['nf'],16,4,6)
        self.anticonv4=nn.ConvTranspose2d(params['nf'],params['nc'],16,4,6)
        
    def forward(self,x):
        # x=F.batchnorm2d(x)
        x=self.norm(x)
        x=F.leaky_relu(self.conv1(x))
        x=F.leaky_relu(self.conv2(x))
        x=F.leaky_relu(self.conv3(x))
        x=F.leaky_relu(self.conv4(x))
        

        x=F.leaky_relu(self.anticonv1(x))
        x=F.leaky_relu(self.anticonv2(x))
        x=F.leaky_relu(self.anticonv3(x))
        x=F.leaky_relu(self.anticonv4(x))
        return x

