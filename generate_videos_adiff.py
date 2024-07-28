import argparse
import torch
from diffusers.models import MotionAdapter
from diffusers import AnimateDiffPipeline, DDIMScheduler
from diffusers.schedulers.scheduling_bdia_ddim import BDIADDIMScheduler
from diffusers.utils import export_to_video
import os
import re


def generate_video_name(prompt, scheduler_type):
    # Extract first few words from the prompt
    words = re.findall(r'\w+', prompt.lower())
    name = '_'.join(words[:3])  # Use first 3 words
    return f"{name}_{scheduler_type}"


def generate_video(prompt, negative_prompt, num_inference_steps, guidance_scale, num_frames, seed,
                   scheduler_type="ddim", gamma=0.5):
    torch.manual_seed(seed)
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
    model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
    if scheduler_type == "ddim":
        scheduler = DDIMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            clip_sample=False,
            timestep_spacing="linspace",
            beta_schedule="linear",
            steps_offset=1,
        )
    elif scheduler_type == "bdia-ddim":
        scheduler = BDIADDIMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            clip_sample=False,
            timestep_spacing="linspace",
            beta_schedule="linear",
            steps_offset=1,
            gamma=gamma
        )
    else:
        raise ValueError("Invalid scheduler_type. Choose 'ddim' or 'bdia-ddim'.")

    pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=torch.float16)
    pipe.scheduler = scheduler
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()

    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_frames=num_frames,
        generator=torch.Generator("cpu").manual_seed(seed),
    )
    frames = output.frames[0]

    output_folder = "./videos_adiff/"
    os.makedirs(output_folder, exist_ok=True)

    video_name = generate_video_name(prompt, scheduler_type)
    video_path = export_to_video(frames, output_video_path=os.path.join(output_folder, f"{video_name}.mp4"))
    return video_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate animated videos using DDIM and BDIA-DDIM schedulers")
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt for video generation')
    parser.add_argument('--negative_prompt', type=str, default="bad quality, worse quality",
                        help='Negative prompt for video generation')
    parser.add_argument('--num_inference_steps', type=int, default=40, help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Guidance scale')
    parser.add_argument('--num_frames', type=int, default=24, help='Number of frames')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--gamma', type=float, default=0.5, help='Gamma value for BDIA-DDIM scheduler')

    args = parser.parse_args()

    # Generate video with DDIM scheduler
    ddim_video_path = generate_video(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_frames=args.num_frames,
        seed=args.seed,
        scheduler_type="ddim"
    )

    # Generate video with BDIA-DDIM scheduler
    bdia_ddim_video_path = generate_video(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_frames=args.num_frames,
        seed=args.seed,
        scheduler_type="bdia-ddim",
        gamma=args.gamma
    )

    print(f"DDIM video saved to: {ddim_video_path}")
    print(f"BDIA-DDIM video saved to: {bdia_ddim_video_path}")