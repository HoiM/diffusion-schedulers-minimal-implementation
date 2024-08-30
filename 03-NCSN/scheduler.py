import math
import torch


class NCSNScheduler(torch.nn.Module):
    def __init__(self, sigma_max=1, sigma_min=0.01, num_timesteps=10):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.num_steps_each_timestep = 100
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.epsilon = 2e-5
        sigmas = torch.exp(
            torch.linspace(
                start=math.log(sigma_max),
                end=math.log(sigma_min),
                steps=num_timesteps
            )
        )
        self.register_buffer("sigmas", sigmas)

    def add_noise(self, images, timesteps, noises=None):
        selected_sigmas = self.sigmas[timesteps].reshape([-1, 1, 1, 1])
        noises = torch.randn_like(images) if noises is None else noises
        noisy_images = images + selected_sigmas * noises
        target_scores = -1 / (selected_sigmas ** 2) * (noisy_images - images)
        lambdas = selected_sigmas ** 2
        return noisy_images, target_scores, lambdas

    def generate(self, model, h, w, batch_size):
        # Anneal Langevin Dynamics
        with torch.no_grad():
            device = self.sigmas.device
            x = torch.rand([batch_size, 1, h, w]).to(torch.float32).to(device)
            for timestep, sigmas_i in enumerate(self.sigmas):
                timestep = torch.tensor([timestep] * batch_size).to(torch.long).to(device)
                alphas_i = self.epsilon * (sigmas_i ** 2) / (self.sigma_min ** 2)
                for t in range(self.num_steps_each_timestep):
                    z_t = torch.randn_like(x)
                    pred_score = model(x, timestep)
                    x = x + alphas_i / 2 * pred_score + (alphas_i ** 0.5) * z_t
            return x
