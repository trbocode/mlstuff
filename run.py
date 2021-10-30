import utils as ut
import autoencoder as ae

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net=ae.Net()
net.to(device)
optimizer=ae.optimizer
criterion=ae.criterion
dataloader=ut.get_data(bsize=3)

for epoch in range(2):
    for i,data in enumerate(dataloader,0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs=net(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
