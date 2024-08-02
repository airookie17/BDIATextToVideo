from .scheduling_ddim import DDIMScheduler, DDIMSchedulerOutput
import torch

class BDIADDIMScheduler(DDIMScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_last = None  # Tracks the last sample
        self.t_last = None  # Tracks the last timestep
        self.gamma = kwargs.get('gamma', 0.5)  # Default gamma value

    def step(self, model_output, timestep, sample, return_dict=True):
        # Find the index of the current timestep
        step_index = torch.where(self.timesteps == timestep)[0].item()
        # Determine the previous timestep
        prev_timestep = self.timesteps[step_index + 1] if step_index < len(self.timesteps) - 1 else 0

        # Compute alpha and sigma values for the current and previous timesteps
        alpha_prod_t = self.alphas_cumprod[timestep].to(sample.device)
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep].to(sample.device) if prev_timestep >= 0 else self.final_alpha_cumprod.to(sample.device)
        beta_prod_t = 1 - alpha_prod_t
        sigma_t = beta_prod_t ** 0.5
        sigma_t_prev = (1 - alpha_prod_t_prev) ** 0.5

        # Predict the original sample (x_0)
        pred_original_sample = (sample - sigma_t * model_output) / (alpha_prod_t ** 0.5)

        # Implement the BDIA-DDIM update logic
        if self.x_last is not None:
            # Compute alpha_prod_t_last for the previous timestep (a_last)
            alpha_prod_t_last = self.alphas_cumprod[self.timesteps[step_index + 1]].to(sample.device) if step_index < len(self.timesteps) - 1 else alpha_prod_t
            sigma_t_last = (1 - alpha_prod_t_last) ** 0.5

            # BDIA-DDIM update step
            prev_sample = (
                self.gamma * self.x_last
                - self.gamma * (alpha_prod_t_last ** 0.5 * pred_original_sample + sigma_t_last * model_output)
                + alpha_prod_t_prev ** 0.5 * pred_original_sample + sigma_t_prev * model_output
            )
        else:
            # Standard DDIM update if no previous sample is available
            prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + sigma_t_prev * model_output

        # Update the state for the next step
        self.x_last = sample
        self.t_last = timestep

        if not return_dict:
            return prev_sample,

        return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

    def set_timesteps(self, num_inference_steps, device="cpu"):
        super().set_timesteps(num_inference_steps, device)
        self.x_last = None
        self.t_last = None
