
import argparse
import torch
from diffusers.models import MotionAdapter
from diffusers import AnimateDiffPipeline, DDIMScheduler
from diffusers.schedulers.scheduling_bdia_ddim import BDIADDIMScheduler
from diffusers.utils import export_to_video
import os

def generate_video(prompt, negative_prompt, num_inference_steps, guidance_scale, num_frames, seed, scheduler_type="ddim", gamma=0.5, output_folder="", video_name=""):
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
    os.makedirs(output_folder, exist_ok=True)
    video_path = export_to_video(frames, output_video_path=os.path.join(output_folder, f"{video_name}.mp4"))

    return video_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate animated video using diffusers library")
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt for video generation')
    parser.add_argument('--negative_prompt', type=str, required=True, help='Negative prompt for video generation')
    parser.add_argument('--num_inference_steps', type=int, required=True, help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, required=True, help='Guidance scale')
    parser.add_argument('--num_frames', type=int, required=True, help='Number of frames')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--scheduler_type', type=str, required=True, choices=['ddim', 'bdia-ddim'], help='Scheduler type')
    parser.add_argument('--gamma', type=float, default=0.5, help='Gamma value for BDIA-DDIM scheduler')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder to save the video')
    parser.add_argument('--video_name', type=str, required=True, help='Name of the output video file')

    args = parser.parse_args()

    video_path = generate_video(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_frames=args.num_frames,
        seed=args.seed,
        scheduler_type=args.scheduler_type,
        gamma=args.gamma,
        output_folder=args.output_folder,
        video_name=args.video_name
    )

    print(f"Video saved to: {video_path}")
