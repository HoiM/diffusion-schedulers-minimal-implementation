import os
import torch
import torchvision
from common.model import get_unet
from scheduler import RectifiedFlowScheduler


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
    num_training_steps = 1000
    batch_size = 1024
    num_epochs = 5000
    lr = 2e-4
    weight_decay = 0
    num_infer_steps = 10

    # data loader
    dataset = torchvision.datasets.MNIST(
        root="common/mnist",
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize([32, 32], antialias=True),
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

    # model
    unet = get_unet(num_training_steps).to(device)
    scheduler = RectifiedFlowScheduler(num_training_steps).to(device)
    
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
            x_1 = images.to(device)
            time_steps = torch.randint(0, num_training_steps, [batch_size]).to(device)
            x_0 = torch.randn_like(x_1)
            x_t = scheduler.add_noise(x_1, x_0, time_steps)
            pred_velocity = unet(x_t, time_steps)
            target_velocity = x_1 - x_0
            loss = loss_func(target_velocity, pred_velocity)
            loss.backward()
            grad_norm = get_grad_norm(unet)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 0.1)
            optimizer.step()
            
            # logging
            global_step += 1
            loss_values.append(loss.item())
            optim_lr = get_lr(optimizer)
            print("Epoch %d/%d Iteration %d/%d: loss=%.6f lr=%.6f grad_norm=%.6f" %
                  ((epoch + 1), num_epochs, (iteration + 1), len(dataloader), loss.item(), optim_lr, grad_norm))

        avg_loss = sum(loss_values) / len(loss_values)
        print("Epoch %d/%d finished. Avg Loss: %.6f" % (epoch + 1, num_epochs, avg_loss))
        if (epoch + 1) % 20 == 0:
            torch.save(unet.state_dict(), "results/01/params_%06d.pth" % (epoch + 1))
            os.system(
                "python test.py %s %d %d %s" % (
                    "results/01/params_%06d.pth" % (epoch + 1),
                    num_infer_steps,
                    seed,
                    "results/01/val_%06d.png" % (epoch + 1)
                )
            )


if __name__ == '__main__':
    train()
