import torch
import numpy as np
from .scheduling_ddim import DDIMScheduler, DDIMSchedulerOutput
from typing import Optional, Union, Tuple

class BDIADDIMScheduler(DDIMScheduler):
    def __init__(self, *args, **kwargs):
        self.gamma = kwargs.pop('gamma', 0.5)  # Default gamma value
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
        generator=None,
        variance_noise: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
    ) -> Union[DDIMSchedulerOutput, Tuple]:
        # Get the device from the sample tensor
        device = sample.device
    
        # Get the index in the timestep array
        step_index = (self.timesteps == timestep).nonzero().item()
        prev_timestep = self.timesteps[step_index + 1] if step_index < len(self.timesteps) - 1 else 0
    
        # Compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep].to(device)
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep].to(device) if prev_timestep >= 0 else self.final_alpha_cumprod.to(device)
        beta_prod_t = 1 - alpha_prod_t
    
        # Compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        else:
            raise ValueError(f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or `v_prediction`")
    
        # 4. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(-self.config.clip_sample_range, self.config.clip_sample_range)
    
        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep).to(device)
        std_dev_t = eta * variance ** (0.5)
    
        if use_clipped_model_output:
            # the pred_original_sample is always re-derived from the clipped x_0 in Glide
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    
        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output
    
        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.x_last is not None:
            a_last = self.alphas_cumprod[self.t_last].to(device)
            x_prev = (
                self.x_last
                - (1 - self.gamma) * (self.x_last - sample)
                - self.gamma * (a_last**0.5 * pred_original_sample + (1 - a_last)**0.5 * model_output - sample)
                + alpha_prod_t_prev ** (0.5) * pred_original_sample
                + pred_sample_direction
                - sample
            )
        else:
            x_prev = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
    
        if eta > 0:
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure that either `generator` or `variance_noise` stays `None`."
                )
    
            if variance_noise is None:
                variance_noise = torch.randn_like(model_output, generator=generator, device=device)
            variance = std_dev_t * variance_noise
    
            x_prev = x_prev + variance
    
        # Update last sample and timestep
        self.x_last = sample.to(device)
        self.t_last = timestep
    
        if not return_dict:
            return (x_prev,)
    
        return DDIMSchedulerOutput(prev_sample=x_prev, pred_original_sample=pred_original_sample)
