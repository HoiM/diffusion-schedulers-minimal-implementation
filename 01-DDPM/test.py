import sys
import torch
import torchvision
from common.model import get_unet
from scheduler import DDPMScheduler


# usage:
# python test.py path/to/ckpt.pth random_number saved_image_path.png


torch.manual_seed(sys.argv[2])
device = torch.device("cuda:0")
num_train_timesteps = 250
unet = get_unet(num_train_timesteps)
unet.eval()
unet.load_state_dict(torch.load(sys.argv[1]))
unet = unet.to(device)
scheduler = DDPMScheduler(0.0001, 0.02, num_train_timesteps).to(device)
images = scheduler.generate(unet, 32, 32, 64)
torchvision.utils.save_image(images * 0.5 + 0.5, sys.argv[3])
