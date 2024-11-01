import copy
import torch


class RectifiedFlowScheduler(torch.nn.Module):
    def __init__(self, num_training_steps):
        super().__init__()
        self.num_training_steps = num_training_steps

    def add_noise(self, x_1, x_0, timestep):
        # x_0 is pure unit normal noise, x_1 is the real image
        t = timestep / self.num_training_steps
        t = t.reshape([t.shape[0], 1, 1, 1])
        x_t = x_1 * t + (1.0 - t) * x_0
        return x_t

    def generate_euler(self, model, h, w, batch_size, num_infer_steps, return_input_noise=False):
        with torch.no_grad():
            device = next(model.parameters()).device
            x_t = torch.randn([batch_size, 1, h, w]).to(device)
            rand_noise = copy.deepcopy(x_t)
            step_size = 1.0 / num_infer_steps
            inference_time_steps = range(0, self.num_training_steps, int(self.num_training_steps / num_infer_steps))
            for timestep in inference_time_steps:
                timestep = torch.ones([batch_size, ]).to(torch.long).to(device) * timestep
                pred_velocity = model(x_t, timestep)
                x_t = x_t = x_t + pred_velocity * step_size
        if return_input_noise:
            return x_t, rand_noise
        else:
            return x_t

    def generate_heun(self, model, h, w, batch_size, num_infer_steps, return_input_noise=False):
        with torch.no_grad():
            device = next(model.parameters()).device
            x_t = torch.randn([batch_size, 1, h, w]).to(device)
            rand_noise = copy.deepcopy(x_t)
            step_size = 1.0 / num_infer_steps
            inference_time_steps = list(range(0, self.num_training_steps, int(self.num_training_steps / num_infer_steps)))
            for i in range(len(inference_time_steps) - 1):
                y_n = x_t
                t_n = torch.ones([batch_size, ]).to(torch.long).to(device) * inference_time_steps[i]
                t_n_1 = torch.ones([batch_size, ]).to(torch.long).to(device) * inference_time_steps[i + 1]
                s1 = model(y_n, t_n)
                y_n_1 = y_n + step_size * s1
                s2 = model(y_n_1, t_n_1)
                y_n_1 = y_n + step_size * (s1 + s2) / 2.0
                x_t = y_n_1
            # last step, use euler
            timestep = inference_time_steps[-1]
            timestep = torch.ones([batch_size, ]).to(torch.long).to(device) * timestep
            pred_velocity = model(x_t, timestep)
            x_t = x_t = x_t + pred_velocity * step_size
        if return_input_noise:
            return x_t, rand_noise
        else:
            return x_t

