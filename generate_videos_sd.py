import argparse
import torch
import os
from diffusers import DiffusionPipeline, DDIMScheduler
from diffusers.schedulers.scheduling_bdia_ddim import BDIADDIMScheduler
from diffusers.utils import export_to_video

def generate_video(prompt, num_inference_steps, num_frames, seed, scheduler_type="ddim", gamma=0.5, output_folder="", video_name=""):
    # Set the seed for reproducibility
    torch.manual_seed(seed)

    # Load the pipeline
    pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16)

    # Set the scheduler
    if scheduler_type == "ddim":
        scheduler = DDIMScheduler.from_pretrained("damo-vilab/text-to-video-ms-1.7b", subfolder="scheduler")
    elif scheduler_type == "bdia-ddim":
        ddim_scheduler = DDIMScheduler.from_pretrained("damo-vilab/text-to-video-ms-1.7b", subfolder="scheduler")
        scheduler = BDIADDIMScheduler.from_config(ddim_scheduler.config, gamma=gamma)
    else:
        raise ValueError("Invalid scheduler_type. Choose 'ddim' or 'bdia-ddim'.")

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate videos using diffusion models.")
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt for video generation')
    parser.add_argument('--num_inference_steps', type=int, required=True, help='Number of inference steps')
    parser.add_argument('--num_frames', type=int, required=True, help='Number of frames')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--scheduler_type', type=str, required=True, choices=['ddim', 'bdia-ddim'], help='Scheduler type')
    parser.add_argument('--gamma', type=float, default=0.5, help='Gamma value for BDIA-DDIM scheduler')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder to save the video')
    parser.add_argument('--video_name', type=str, required=True, help='Name of the output video file')

    args = parser.parse_args()

    video_path = generate_video(
        prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        num_frames=args.num_frames,
        seed=args.seed,
        scheduler_type=args.scheduler_type,
        gamma=args.gamma,
        output_folder=args.output_folder,
        video_name=args.video_name
    )

    print(f"Video saved to: {video_path}")
