import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset
import os
from PIL import Image

root = '/home/ML_courses/03683533_2021/dataset/'


class MyDataset():
    
    def _init_(self, image_paths):
        self.image_paths = os.listdir(root)

    def _getitem_(self, index):
        image = Image.open(root + self.image_paths[index])

        transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.5], [0.5])  
        ])

        x = transform(image)
        return x, x # The label is the original pic

    def _len_(self):
        return len(self.image_paths)



def get_pics(params):
    #Preprocessing will go here
    
    dataset=MyDataset()
    
    dataloader=torch.utils.data.DataLoader(dataset=dataset,batch_size=params['bsize'],shuffle=True)
    return dataloader


