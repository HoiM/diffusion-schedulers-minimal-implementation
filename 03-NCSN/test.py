import sys
import torch
import torchvision
from common.model import get_unet
from scheduler import NCSNScheduler


# usage:
# python test.py path/to/ckpt.pth random_number saved_image_path.png


torch.manual_seed(int(sys.argv[2]))
device = torch.device("cuda:0")
sigma_max = 1.0
sigma_min = 0.01
num_train_timesteps = 10
unet = get_unet(num_train_timesteps)
unet.eval()
unet.load_state_dict(torch.load(sys.argv[1]))
unet = unet.to(device)
scheduler = NCSNScheduler(sigma_max, sigma_min, num_train_timesteps).to(device)
images = scheduler.generate(unet, 32, 32, 64)
torchvision.utils.save_image(images, sys.argv[3])
