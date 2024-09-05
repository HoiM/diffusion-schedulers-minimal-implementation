import math
import torch


class SDEVEPCScheduler(torch.nn.Module):
    def __init__(self, sigma_max=1, sigma_min=0.01, num_timesteps=10, num_corrector_steps=100, snr=0.15):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.num_steps_each_timestep = 100
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.num_corrector_steps = num_corrector_steps
        self.snr = snr
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
        # SDE: Variation Exploding with Predictor-Corrector
        with torch.no_grad():
            device = self.sigmas.device
            sigmas = [m for m in self.sigmas]
            sigmas.append(0.0)
            x = torch.rand([batch_size, 1, h, w]).to(torch.float32).to(device)
            for i in range(len(self.sigmas) - 1):
                timestep = torch.ones([batch_size, ], dtype=torch.long, device=device) * i
                # predict
                curr_sigma = sigmas[i]
                next_sigma = sigmas[i + 1]
                score = model(x, timestep)
                diff_sq_sigma = curr_sigma ** 2 - next_sigma ** 2
                rand_noise = torch.randn_like(x)
                x = x + diff_sq_sigma * score + diff_sq_sigma ** 0.5 * rand_noise
                # correct
                for _ in range(self.num_corrector_steps):
                    score = model(x, timestep)
                    rand_noise = torch.randn_like(x)
                    # calculate epsilon (step size)
                    noise_norm = torch.norm(rand_noise.reshape([batch_size, -1])).mean()
                    score_norm = torch.norm(score.reshape([batch_size, -1])).mean()
                    step_size = (self.snr * noise_norm / score_norm) ** 2 * 2
                    # update with score
                    x = x + step_size * score + (step_size * 2) ** 0.5 * rand_noise


            return x
