import argparse
import torch
import os
from diffusers import DiffusionPipeline, DDIMScheduler
from diffusers.schedulers.scheduling_bdia_ddim import BDIADDIMScheduler
from diffusers.utils import export_to_video

def generate_video(scheduler, prompt, num_frames, num_inference_steps, seed, output_folder, video_name):
    # Set the seed for reproducibility
    torch.manual_seed(seed)

    # Load the pipeline
    pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16)

    # Set the scheduler
    pipe.scheduler = scheduler

    pipe.enable_model_cpu_offload()
    pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
    pipe.enable_vae_slicing()

    # Generate the initial latents
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latents = torch.randn(
        (1, 4, num_frames, 64, 64),
        generator=generator,
        device="cpu",
        dtype=torch.float16
    )

    # Generate video frames
    video_frames = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
        latents=latents
    ).frames[0]

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Export the video
    video_path = export_to_video(video_frames, output_video_path=os.path.join(output_folder, f"{video_name}.mp4"))

    return video_path

def process_prompts(prompts, num_frames, num_inference_steps, seed, gamma=1.0):
    # Create the base output folder
    base_output_folder = "./videos_sd/"
    os.makedirs(base_output_folder, exist_ok=True)

    # Create subfolders for each scheduler
    ddim_folder = os.path.join(base_output_folder, "ddim")
    bdia_ddim_folder = os.path.join(base_output_folder, "bdia_ddim")
    os.makedirs(ddim_folder, exist_ok=True)
    os.makedirs(bdia_ddim_folder, exist_ok=True)

    # Initialize schedulers
    ddim_scheduler = DDIMScheduler.from_pretrained("damo-vilab/text-to-video-ms-1.7b", subfolder="scheduler")
    bdia_ddim_scheduler = BDIADDIMScheduler.from_config(ddim_scheduler.config, gamma=gamma)

    for i, prompt in enumerate(prompts):
        # Generate video with DDIMScheduler
        ddim_path = generate_video(
            ddim_scheduler,
            prompt,
            num_frames,
            num_inference_steps,
            seed,
            ddim_folder,
            f"ddim_{i+1}"
        )
        print(f"DDIM video {i+1} saved to: {ddim_path}")

        # Generate video with BDIADDIMScheduler
        bdia_ddim_path = generate_video(
            bdia_ddim_scheduler,
            prompt,
            num_frames,
            num_inference_steps,
            seed,
            bdia_ddim_folder,
            f"bdia_{i+1}"
        )
        print(f"BDIA-DDIM video {i+1} saved to: {bdia_ddim_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate videos using diffusion models.")

    parser.add_argument("--prompts", type=str, nargs='+', required=True, help="List of prompts for video generation.")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames in the video.")
    parser.add_argument("--num_inference_steps", type=int, default=40, help="Number of inference steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma value for the BDIA-DDIM scheduler.")

    args = parser.parse_args()

    process_prompts(args.prompts, args.num_frames, args.num_inference_steps, args.seed, args.gamma)
