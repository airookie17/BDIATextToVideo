import torch
import numpy as np
from .scheduling_ddim import DDIMScheduler, DDIMSchedulerOutput
from typing import Optional, Union, Tuple

class BDIADDIMScheduler(DDIMScheduler):
    def __init__(self, *args, **kwargs):
        self.gamma = kwargs.pop('gamma', 0.5)  # Default gamma to 0.5
        super().__init__(*args, **kwargs)
        self.x_last = None
        self.t_last = None

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        super().set_timesteps(num_inference_steps, device)
        self.timesteps = self.timesteps.to(device)
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
        step_index = (self.timesteps == timestep).nonzero().item()
        prev_timestep = self.timesteps[step_index + 1] if step_index < len(self.timesteps) - 1 else 0

        # Extract relevant alphas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        # Compute predicted original sample
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        else:
            raise ValueError(f"Unsupported prediction type: {self.config.prediction_type}")

        # Clip or threshold predicted sample
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(-self.config.clip_sample_range, self.config.clip_sample_range)

        # Compute the previous noisy sample x_t -> x_t-1
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        
        if self.x_last is not None:
            a_last = self.alphas_cumprod[self.t_last]
            x_prev = (
                self.x_last
                - (1 - self.gamma) * (self.x_last - sample)
                - self.gamma * (a_last ** 0.5 * pred_original_sample + (1 - a_last) ** 0.5 * model_output - sample)
                + alpha_prod_t_prev ** 0.5 * pred_original_sample
                + pred_sample_direction
                - sample
            )
        else:
            x_prev = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction

        # Add noise
        if eta > 0:
            noise = torch.randn_like(model_output, generator=generator)
            variance = self._get_variance(timestep, prev_timestep)
            noise = eta * variance ** 0.5 * noise
            x_prev = x_prev + noise

        # Update last sample and timestep
        self.x_last = sample.to(sample.device)
        self.t_last = timestep

        if not return_dict:
            return (x_prev,)

        return DDIMSchedulerOutput(prev_sample=x_prev, pred_original_sample=pred_original_sample)

    def add_noise(self, original_samples, noise, timesteps):
        return super().add_noise(original_samples, noise, timesteps)

    def __len__(self):
        return self.config.num_train_timesteps
