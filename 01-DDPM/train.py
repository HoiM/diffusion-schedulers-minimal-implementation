import os
import copy
import torch
import torchvision
from common.model import get_unet
from scheduler import DDPMScheduler


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def ema_update(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1.0 - decay))


def train():
    # hyper-parameters
    seed = 97823
    torch.manual_seed(seed)
    device = torch.device("cuda:0")
    num_train_timesteps = 250
    batch_size = 256
    num_epochs = 20
    lr = 1e-4
    weight_decay = 0
    ema_decay = 0.999

    # data loader
    dataset = torchvision.datasets.MNIST(
        root="common/mnist",
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize([32, 32]),
            torchvision.transforms.Normalize(0.5, 0.5)
        ])
    )
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    total_train_steps = len(dataloader) * num_epochs
    
    # model
    unet = get_unet(num_train_timesteps).to(device)
    ema_unet = copy.deepcopy(unet)
    scheduler = DDPMScheduler(0.0001, 0.02, num_train_timesteps).to(device)
    
    # loss
    loss_func = torch.nn.MSELoss().to(device)
    
    # optimizer
    optimizer = torch.optim.AdamW(
        params=unet.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # train
    global_step = 0
    for epoch in range(num_epochs):
        loss_values = list()
        for iteration, (images, _) in enumerate(dataloader):
            # train one step
            optimizer.zero_grad()
            x_0 = images.to(device)
            timesteps = torch.randint(low=0, high=num_train_timesteps, size=[batch_size]).to(device)
            unit_normal_noise = torch.randn_like(x_0)
            x_t = scheduler.add_noise(x_0, timesteps, unit_normal_noise)
            pred_eps = unet(x_t, timesteps)
            loss = loss_func(pred_eps, unit_normal_noise)
            loss.backward()
            grad_norm = get_grad_norm(unet)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 0.1)
            optimizer.step()
            ema_update(unet, ema_unet, ema_decay)
            
            # logging
            global_step += 1
            loss_values.append(loss.item())
            optim_lr = get_lr(optimizer)
            print("Epoch %d/%d Iteration %d/%d: loss=%.6f lr=%.6f grad_norm=%.6f" %
                  ((epoch + 1), num_epochs, (iteration + 1), len(dataloader), loss.item(), optim_lr, grad_norm))

        avg_loss = sum(loss_values) / len(loss_values)
        print("Epoch %d/%d finished. Avg Loss: %.6f" % (epoch + 1, num_epochs, avg_loss))
        if (epoch + 1) % 2 == 0:
            torch.save(ema_unet.state_dict(), "results/params_%02d.pth" % (epoch + 1))
            os.system("python test.py %s %d %s" % ("results/params_%02d.pth" % (epoch + 1), seed, "results/val_%06d.png" % (epoch + 1)))


if __name__ == '__main__':
    train()

