import utils as ut
import autoencoder as ae
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

params = {'bsize': 128,
         'nc': 3,
         'nf': 64,
         'ncode': 4,
         'n_epochs': 5}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net=ae.Net(params).to(device)
optimizer=optim.Adam(net.parameters(),lr=0.001)
criterion=torch.nn.MSELoss()
dataloader=ut.get_pics(params)
alllosses=[]
for epoch in range(params['n_epochs']):
    for i,data in enumerate(dataloader,0):
        if (i<8):
            continue
        inputs = data[0].to(device)
        inputs2=inputs
        optimizer.zero_grad()
        outputs=net(inputs)
        loss=criterion(outputs,inputs2)
        loss.backward()
        optimizer.step()
        alllosses.append(loss.item())
    print(loss.item())
torch.save({
    'autoencoder' : net.state_dict(),
    'optimizer':optimizer.state_dict(),
    'params': params
    }, 'model/modelf_111.pth')


plt.plot(alllosses)
plt.savefig("model111.png")
plt.show()
