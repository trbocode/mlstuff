import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset

root = '/home/ML_courses/03683533_2021/dataset'

def get_pics(params):
    #Preprocessing will go here
    transform=transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize([0.5], [0.5])  
        ])

    dataset=dset.ImageFolder(root=root,transform=transform)
    
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=params['bsize'],shuffle=True)
    return dataloader
