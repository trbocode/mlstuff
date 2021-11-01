import utils as ut
import autoencoder as ae
import torch

params = {'bsize': 128,
          'n_epochs': 5}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net=ae.Net()
net.to(device)
optimizer=ae.optimizer
criterion=ae.criterion
dataloader=ut.get_pics(params)

for epoch in range(params['n_epochs']):
    for i,data in enumerate(dataloader,0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs=net(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
