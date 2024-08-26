import sys
import torch
import torchvision
from common.model import get_unet
from scheduler import DDIMScheduler


"""
Usage:
    python sample.py ../01-DDPM/results/params_20.pth 3241432 ./results.png
Input arguments:
    1, path to the params trained with DDPM
    2, a random seed
    3, path to save the generated result
"""

torch.manual_seed(sys.argv[2])
device = torch.device("cuda:0")
num_train_timesteps = 250
unet = get_unet(num_train_timesteps)
unet.eval()
unet.load_state_dict(torch.load(sys.argv[1]))
unet = unet.to(device)
scheduler = DDIMScheduler(0.0001, 0.02, num_train_timesteps).to(device)
images = scheduler.generate(unet, 32, 32, 64, 50, 1, device)
torchvision.utils.save_image(images * 0.5 + 0.5, sys.argv[3])
