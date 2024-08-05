from .scheduling_ddim import DDIMScheduler, DDIMSchedulerOutput
import torch
import numpy as np
from typing import Optional, Union, Tuple, List


class BDIADDIMScheduler(DDIMScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_last = None  # Tracks the last sample
        self.t_last = None  # Tracks the last timestep
        self.gamma = kwargs.get('gamma', 0.5)  # Default gamma value

    def set_timesteps(self, num_inference_steps: int):
        self.num_inference_steps = num_inference_steps
        self.timesteps = np.arange(0, self.num_train_timesteps, self.num_train_timesteps // self.num_inference_steps)[::-1]
        self.timesteps = torch.tensor(self.timesteps, device=device)
        self.x_last = None
        self.t_last = None

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator: Optional[torch.Generator] = None,
        variance_noise: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
    ) -> Union[DDIMSchedulerOutput, Tuple]:
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output

        if self.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)

        variance = 0
        if eta > 0:
            variance = self._get_variance(timestep, prev_timestep)
            if variance_noise is None:
                variance_noise = torch.randn_like(model_output, generator=generator)
            variance = eta * variance**0.5 * variance_noise

        if use_clipped_model_output:
            model_output = (sample - alpha_prod_t**0.5 * pred_original_sample) / beta_prod_t**0.5
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        if self.x_last is not None:
            a_last = self.alphas_cumprod[timestep + 1] if timestep + 1 < len(self.alphas_cumprod) else self.final_alpha_cumprod
            x_prev = (self.x_last - (1 - self.gamma) * (self.x_last - sample)
                      - self.gamma * (a_last**0.5 * pred_original_sample + (1 - a_last)**0.5 * model_output - sample)
                      + alpha_prod_t_prev**0.5 * pred_original_sample + (1 - alpha_prod_t_prev - variance)**0.5 * model_output - sample)
        else:
            x_prev = alpha_prod_t_prev**0.5 * pred_original_sample + (1 - alpha_prod_t_prev - variance)**0.5 * model_output + variance

        self.x_last = sample
        self.t_last = timestep

        if not return_dict:
            return (x_prev,)

        return DDIMSchedulerOutput(prev_sample=x_prev, pred_original_sample=pred_original_sample)

    def reset(self):
        self.x_last = None
        self.t_last = None

