from .scheduling_ddim import DDIMScheduler, DDIMSchedulerOutput

class BDIADDIMScheduler(DDIMScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_last = None
        self.t_last = None
        self.gamma = kwargs.get('gamma', 0.5)  # Default gamma value

    def step(self, model_output, timestep, sample, return_dict=True):
        step_index = (self.timesteps == timestep).nonzero().item()
        prev_timestep = self.timesteps[step_index + 1] if step_index < len(self.timesteps) - 1 else 0

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * model_output
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        # BDIA modifications
        if self.x_last is not None:
            alpha_prod_t_last = self.alphas_cumprod[self.t_last] if self.t_last is not None else alpha_prod_t
            prev_sample = (
                self.x_last
                - (1 - self.gamma) * (self.x_last - sample)
                - self.gamma * (alpha_prod_t_last ** 0.5 * pred_original_sample + (1 - alpha_prod_t_last) ** 0.5 * model_output - sample)
                + alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction - sample
            )

        self.x_last = sample
        self.t_last = timestep

        if not return_dict:
            return (prev_sample,)

        return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

    def set_timesteps(self, num_inference_steps, device="cpu"):
        super().set_timesteps(num_inference_steps, device)
        self.x_last = None
        self.t_last = None
