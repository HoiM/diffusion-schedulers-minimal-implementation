import sys
import torch
import torchvision
from common.model import get_unet
from scheduler import RectifiedFlowScheduler


# usage:
# python test.py path/to/ckpt.pth num_infer_steps random_number saved_image_path.png

param_path = sys.argv[1]
num_infer_steps = int(sys.argv[2])
rand_seed = int(sys.argv[3])
image_save_path = sys.argv[4]

torch.manual_seed(rand_seed)
device = torch.device("cuda:0")
num_training_steps = 1000
unet = get_unet(num_training_steps)
unet.eval()
unet.load_state_dict(torch.load(param_path))
unet = unet.to(device)
scheduler = RectifiedFlowScheduler(num_training_steps).to(device)
images = scheduler.generate_euler(unet, 32, 32, 64, num_infer_steps)
#images = scheduler.generate_heun(unet, 32, 32, 64, num_infer_steps)
torchvision.utils.save_image(images * 0.5 + 0.5, image_save_path)
