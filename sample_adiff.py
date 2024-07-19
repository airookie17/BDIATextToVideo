import subprocess
from assessment import assess_videos
import os


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
    prompt = "a panda eating a bamboo in an enchanted forest, masterpiece, best quality, highlydetailed, ultradetailed"
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
        video_name="panda_eating_bamboo_ddim"
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
        video_name="panda_eating_bamboo_bdia_ddim"
    )

    # Assess the generated videos
    # assessment_results = assess_videos(
    #     video1_path=os.path.join(output_folder, "panda_eating_bamboo_ddim.mp4"),
    #     video2_path=os.path.join(output_folder, "panda_eating_bamboo_bdia_ddim.mp4")
    # )
    #
    # print("Assessment Results:")
    # print(
    #     f"Temporal Consistency - DDIM: {assessment_results['temporal_consistency']['video1']}, BDIA-DDIM: {assessment_results['temporal_consistency']['video2']}")
    # print(f"LPIPS - DDIM: {assessment_results['lpips']['video1']}, BDIA-DDIM: {assessment_results['lpips']['video2']}")
    print("Running VMAF assessment...")
    subprocess.run(['ffmpeg', '-i', os.path.join(output_folder, "panda_eating_bamboo_ddim.mp4"),
                    '-i', os.path.join(output_folder, "panda_eating_bamboo_bdia_ddim.mp4"),
                    '-lavfi', 'libvmaf="model_path=/usr/share/model/vmaf_v0.6.1.pkl"', '-f', 'null', '-'])