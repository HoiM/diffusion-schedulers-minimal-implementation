import os
import copy
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
    batch_size = 256
    num_train_iterations = 1000
    lr = 4e-6
    weight_decay = 0
    num_infer_steps = 1
    pretrained_param = "results/01/params_005000.pth"

    # data loader (model pretrained with rectified flow)
    teacher_unet = get_unet(num_training_steps).to(device)
    teacher_unet.load_state_dict(torch.load(pretrained_param, device))
    teacher_unet.eval()

    # model
    unet = get_unet(num_training_steps).to(device)
    unet.load_state_dict(torch.load(pretrained_param, device))
    unet.train()
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
    num_reflow = 1
    print("%d-reflow starts..." % num_reflow)
    for iteration in range(num_train_iterations):
        with torch.no_grad():
            images, noises = scheduler.generate_euler(unet, 32, 32, batch_size, 10, True)
        # train one step
        x_1 = images.to(device)
        x_0 = noises.to(device)
        for _ in range(1):
            optimizer.zero_grad()
            time_steps = torch.randint(0, num_training_steps, [batch_size]).to(device)
            x_t = scheduler.add_noise(x_1, x_0, time_steps)
            pred_velocity = unet(x_t, time_steps)
            target_velocity = x_1 - x_0
            loss = loss_func(target_velocity, pred_velocity)
            loss.backward()
            grad_norm = get_grad_norm(unet)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 0.01)
            optimizer.step()

        # logging
        optim_lr = get_lr(optimizer)
        print("Iteration %d: loss=%.6f lr=%.6f grad_norm=%.6f" %
                ((iteration + 1), loss.item(), optim_lr, grad_norm))

        if (iteration + 1) % 100 == 0:
            param_save_path = "results/02/params_%06d.pth" % (iteration + 1)
            torch.save(unet.state_dict(), param_save_path)
            os.system(
                "python test.py %s %d %d %s" % (
                    param_save_path,
                    num_infer_steps,
                    seed,
                    "results/02/val_%06d.png" % (iteration + 1)
                )
            )
        if (iteration + 1) % 10000 == 0:
            # update the teacher model, simulating k-th reflow
            teacher_unet = copy.deepcopy(unet)
            teacher_unet.eval()
            num_reflow += 1
            print("%d-reflow starts..." % num_reflow)


if __name__ == '__main__':
    train()
