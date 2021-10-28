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
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=F.relu(self.anticonv1(x))
        x=F.relu(self.anticonv2(x))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net=Net()
net.to(device)
criterion=nn.MSELoss()
optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
for epoch in range(2):
    for i,data in enumerate(inputs,0)
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs=net(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()

