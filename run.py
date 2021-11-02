import utils as ut
import autoencoder as ae
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

params = {'bsize': 128,
         'nc': 3,
         'nf': 256,
         'ncode': 3,
         'n_epochs': 5}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net=ae.Net(params)
net.to(device)
optimizer=optim.Adam(net.parameters(),lr=0.001)
criterion=ae.criterion
dataloader=ut.get_pics(params)
alllosses=[]

for epoch in range(params['n_epochs']):
    for i,data in enumerate(dataloader,0):
        inputs = data[0].to(device)
        realbsize=inputs.size(0)

        optimizer.zero_grad()
        outputs=net(inputs)
        loss=criterion(outputs,inputs)
        loss.backward()
        optimizer.step()
        alllosses.append(loss)
torch.save({
    'autoencoder' : net.state_dict(),
    'optimizer':optimizer.state_dict(),
    'params': params
    }, 'model/modelf.pth')


plt.plot(alllosses)
plt.show()
