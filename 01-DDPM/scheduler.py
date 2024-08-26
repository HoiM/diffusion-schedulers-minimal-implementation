import torch


class DDPMScheduler(torch.nn.Module):
    def __init__(self, beta_min, beta_max, num_steps):
        super().__init__()
        self.num_steps = num_steps
        betas = torch.linspace(beta_min, beta_max, num_steps)
        alphas = torch.ones_like(betas) - betas
        alphas_cumprod = torch.cumprod(alphas, 0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def add_noise(self, x_0, timestep, noise):
        alpha_bar_t = self.alphas_cumprod[timestep]
        alpha_bar_t = alpha_bar_t.reshape(alpha_bar_t.shape[0], 1, 1, 1)
        x_t = alpha_bar_t ** 0.5 * x_0 + (1 - alpha_bar_t) * noise
        return x_t

    def get_loss_weights(self, timestep):
        alpha_t = self.alphas[timestep]
        beta_t = self.betas[timestep]
        alpha_bar_t = self.alphas_cumprod[timestep]
        alpha_bar_t_1 = self.alphas_cumprod[timestep - 1]
        sigma_t_squared = (1 - alpha_bar_t_1) / (1 - alpha_bar_t) * beta_t
        #sigma_t_squared = beta_t
        weights = beta_t ** 2 / (2 * sigma_t_squared * alpha_t * (1 - alpha_bar_t))
        return weights

    def step(self, x_t, pred_eps, timestep, unit_normal_noise=None):
        alpha_t = self.alphas[timestep]
        alpha_t = alpha_t.reshape([alpha_t.shape[0], 1, 1, 1])
        beta_t = self.betas[timestep]
        beta_t = beta_t.reshape([beta_t.shape[0], 1, 1, 1])
        alpha_bar_t = self.alphas_cumprod[timestep]
        alpha_bar_t = alpha_bar_t.reshape([alpha_bar_t.shape[0], 1, 1, 1])
        #alpha_bar_t_1 = self.alphas_cumprod[timestep - 1]
        #alpha_bar_t_1 = alpha_bar_t_1.reshape([alpha_bar_t_1.shape[0], 1, 1, 1])

        #sigma_t_squared = (1 - alpha_bar_t_1) / (1 - alpha_bar_t) * beta_t
        sigma_t_squared = beta_t

        mean = 1 / (alpha_t ** 0.5) * (x_t - (1 - alpha_t) / ((1 - alpha_bar_t) ** 0.5) * pred_eps)
        noise = torch.randn_like(x_t) if unit_normal_noise is None else unit_normal_noise
        var = sigma_t_squared * noise
        x_t_1 = mean + var
        return x_t_1

    def generate(self, model, h, w, batch_size):
        device = self.betas.device
        #all_xs = list()
        with torch.no_grad():
            x_t = torch.randn([batch_size, 1, h, w]).to(device)
            #all_xs.append(x_t)
            for t in reversed(range(0, self.num_steps)):
                t = torch.tensor([t] * batch_size).to(torch.long).to(device)
                pred_eps = model(x_t, t)
                x_t_1 = self.step(x_t, pred_eps, t)
                x_t = x_t_1
                #all_xs.append(x_t)
            #all_xs = torch.cat(all_xs, 0)
        #return all_xs
        return x_t
