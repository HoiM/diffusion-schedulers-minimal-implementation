import os
import torch
import torchvision
from common.model import get_unet
from scheduler import NCSNScheduler


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


def train():
    # hyper-parameters
    seed = 97823
    torch.manual_seed(seed)
    device = torch.device("cuda:0")
    sigma_max = 1.0
    sigma_min = 0.01
    num_train_timesteps = 10
    batch_size = 256
    num_epochs = 100
    lr = 2e-5
    weight_decay = 0

    # data loader
    dataset = torchvision.datasets.MNIST(
        root="common/mnist",
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize([32, 32]),
            #torchvision.transforms.Normalize(0.5, 0.5)
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

    # model
    unet = get_unet(num_train_timesteps).to(device)
    scheduler = NCSNScheduler(sigma_max, sigma_min, num_train_timesteps).to(device)

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
            x_0 = torch.rand_like(x_0) / 256 + x_0
            timesteps = torch.randint(low=0, high=num_train_timesteps, size=[batch_size]).to(device)
            unit_normal_noise = torch.randn_like(x_0)
            x_t, target_scores, lambdas = scheduler.add_noise(x_0, timesteps, unit_normal_noise)
            pred_scores = unet(x_t, timesteps)
            loss = torch.mean(0.5 * ((pred_scores - target_scores) ** 2) * lambdas)
            loss.backward()
            grad_norm = get_grad_norm(unet)
            optimizer.step()

            # logging
            global_step += 1
            loss_values.append(loss.item())
            optim_lr = get_lr(optimizer)
            print("Epoch %d/%d Iteration %d/%d: loss=%.6f lr=%.6f grad_norm=%.6f" %
                  ((epoch + 1), num_epochs, (iteration + 1), len(dataloader), loss.item(), optim_lr, grad_norm))

        avg_loss = sum(loss_values) / len(loss_values)
        print("Epoch %d/%d finished. Avg Loss: %.6f" % (epoch + 1, num_epochs, avg_loss))
        if (epoch + 1) % 2 == 0:
            torch.save(unet.state_dict(), "results/params_%02d.pth" % (epoch + 1))
            os.system("python test.py %s %d %s" % ("results/params_%02d.pth" % (epoch + 1), seed, "results/val_%06d.png" % (epoch + 1)))


if __name__ == '__main__':
    train()

