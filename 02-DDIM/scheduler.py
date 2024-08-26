import torch


class DDIMScheduler(torch.nn.Module):
    def __init__(self, beta_min, beta_max, num_train_steps):
        super().__init__()
        self.num_train_steps = num_train_steps
        betas = torch.linspace(beta_min, beta_max, num_train_steps)
        alphas = torch.ones_like(betas) - betas
        alphas_cumprod = torch.cumprod(alphas, 0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def step(self, x_t, pred_eps, timestep, prev_timestep, eta, unit_normal_noise=None):
        alpha_bar_t = self.alphas_cumprod[timestep]
        alpha_bar_t = alpha_bar_t.reshape([alpha_bar_t.shape[0], 1, 1, 1])
        alpha_bar_t_1 = self.alphas_cumprod[prev_timestep]
        alpha_bar_t_1 = alpha_bar_t_1.reshape([alpha_bar_t_1.shape[0], 1, 1, 1])

        pred_x_0 = (x_t - (1 - alpha_bar_t) ** 0.5 * pred_eps) / (alpha_bar_t ** 0.5)
        sigma_t = (((1 - alpha_bar_t_1) / (1 - alpha_bar_t)) ** 0.5) * ((1 - alpha_bar_t / alpha_bar_t_1) ** 0.5)
        mean = alpha_bar_t_1 ** 0.5 * pred_x_0 + (1 - alpha_bar_t_1 - sigma_t ** 2) ** 0.5 * pred_eps
        if eta > 0:
            unit_normal_noise = torch.randn_like(x_t) if unit_normal_noise is None else unit_normal_noise
            var = eta * (sigma_t ** 2) * unit_normal_noise
            x_t_1 = mean + var
        else:
            x_t_1 = mean
        return x_t_1

    def get_x0_from_x1(self, x1, pred_eps):
        x0 = (x1 - (1 - self.alphas_cumprod[0]) ** 0.5 * pred_eps) / (self.alphas_cumprod[0] ** 0.5)
        return x0

    def generate(self, model, h, w, batch_size, num_steps, eta, device):
        self.inference_steps = torch.linspace(0, self.num_train_steps - 1, num_steps)
        self.inference_steps = reversed((self.inference_steps + 0.5).to(torch.long))
        with torch.no_grad():
            #all_res = list()
            x_t = torch.randn([batch_size, 1, h, w]).to(device)
            for i, t in enumerate(self.inference_steps[:-1]):
                #all_res.append(x_t)
                t = torch.tensor([t] * batch_size).to(torch.long).to(device)
                t_1 = self.inference_steps[i + 1]
                t_1 = torch.tensor([t_1] * batch_size).to(torch.long).to(device)
                pred_eps = model(x_t, t)
                x_t_1 = self.step(x_t, pred_eps, t, t_1, eta)
                x_t = x_t_1
            #all_res.append(x_t)
            pred_eps = model(x_t, torch.tensor([0] * batch_size).to(torch.long).to(device))
            x0 = self.get_x0_from_x1(x_t, pred_eps)
            #all_res.append(x0)
            #all_res = torch.cat(all_res, 2)
        return x0
