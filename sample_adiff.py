import subprocess


def generate_video(prompt, negative_prompt, num_inference_steps, guidance_scale, num_frames, seed, scheduler_type,
                   gamma, output_folder, video_name):
    command = [
        'python', 'generate_videos_adiff.py',
        '--prompt', prompt,
        '--negative_prompt', negative_prompt,
        '--num_inference_steps', str(num_inference_steps),
        '--guidance_scale', str(guidance_scale),
        '--num_frames', str(num_frames),
        '--seed', str(seed),
        '--scheduler_type', scheduler_type,
        '--output_folder', output_folder,
        '--video_name', video_name
    ]
    if scheduler_type == "bdia-ddim":
        command.extend(['--gamma', str(gamma)])
    subprocess.run(command, check=True)


if __name__ == "__main__":
    # Define video generation parameters
    prompt = "a rabbit eating a carrot in a rainforest, masterpiece, best quality, highlydetailed, ultradetailed"
    negative_prompt = "bad quality, worse quality"
    num_inference_steps = 40
    guidance_scale = 7.5
    seed = 0
    output_folder = "./videos"

    # Generate video with DDIM scheduler
    generate_video(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_frames=24,
        seed=seed,
        scheduler_type="ddim",
        gamma=0.0,  # Gamma is not used in DDIM
        output_folder=output_folder,
        video_name="rabbit_eating_carrot_ddim"
    )

    # Generate video with BDIA-DDIM scheduler
    generate_video(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_frames=24,
        seed=seed,
        scheduler_type="bdia-ddim",
        gamma=1.0,
        output_folder=output_folder,
        video_name="rabbit_eating_carrot_bdia_ddim"
    )
