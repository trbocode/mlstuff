import argparse
import utils as ut
import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import autoencoder as ae

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', default='model/modelf_4.pth', help='Checkpoint to load path from')
parser.add_argument('-num_output', default=64, help='Number of generated outputs')
args = parser.parse_args()

# Load the checkpoint file.
state_dict = torch.load(args.load_path)

# Set the device to run on: GPU or CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
# Get the 'params' dictionary from the loaded state_dict.
params = state_dict['params']

# Create the generator network.
net = ae.Net(params).to(device)
# Load the trained generator weights.
dataloader=ut.get_pics(params)
criterion=torch.nn.MSELoss()
net.load_state_dict(state_dict['autoencoder'])
print(net)

print(args.num_output)

# Turn off gradient calculation to speed up the process.
with torch.no_grad():
	# Get generated image from the other image
    for i,data in enumerate(dataloader,0):
        if(i>0):
            break
        # Display the generated image.
        inputs = data[0].to(device)
        outputs=net(inputs)
        print(criterion(outputs,inputs))
        plt.axis("off")
        plt.title("Given Images")
        plt.imshow(np.transpose(vutils.make_grid(data[0][:64], padding=2, normalize=True).cpu(), (1,2,0)))
        plt.savefig("given.png")
        plt.show()
        # Display the generated image.
        plt.axis("off")
        plt.title("Generated Images")
        plt.imshow(np.transpose(vutils.make_grid(outputs[:64], padding=2, normalize=True).cpu(), (1,2,0)))
        plt.savefig("changed.png")
        plt.show()




##Stolen directly from the DCGAN thing with minimal changes
