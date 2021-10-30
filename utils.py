import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset

root = '/home/ML_courses/03683533_2021/dataset'

def get_pics(params):
    #Preprocessing will go here
    transform=transforms.Compose([
        ])
    dataset=dset.ImageFolder(root=root,transform=none)
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=params['bsize'],shuffle=True)
    return dataloader
