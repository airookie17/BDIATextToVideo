# sample_adiff.py

import subprocess

import cv2
from sentence_transformers import SentenceTransformer
from torchvision.models import inception_v3

from assessment import temporal_consistency, inception_score, prompt_similarity, fid_score


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


def load_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def assess_video(video_path, prompt):
    video = load_video(video_path)

    # Temporal Consistency
    tc_score = temporal_consistency(video)

    # Inception Score
    inception_model = inception_v3(pretrained=True, transform_input=False).eval()
    is_score = inception_score(video, inception_model)

    # Prompt Similarity
    sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    ps_score = prompt_similarity(video, prompt, sentence_model)

    return tc_score, is_score, ps_score


if __name__ == "__main__":
    # Define video generation parameters
    prompt = "A cat wearing glasses reading a book, masterpiece, best quality, highly detailed, ultradetailed"
    negative_prompt = "bad quality, worse quality"
    num_inference_steps = 40
    guidance_scale = 7.5
    seed = 0
    output_folder = "./videos_adiff"

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
        video_name="cat_reading_ddim"
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
        video_name="cat_reading_bdia_ddim"
    )

    # Assess videos
    ddim_path = f"{output_folder}/cat_reading_ddim.mp4"
    bdia_ddim_path = f"{output_folder}/cat_reading_bdia_ddim.mp4"

    ddim_scores = assess_video(ddim_path, prompt)
    bdia_ddim_scores = assess_video(bdia_ddim_path, prompt)

    # Calculate FID score
    ddim_video = load_video(ddim_path)
    bdia_ddim_video = load_video(bdia_ddim_path)
    inception_model = inception_v3(pretrained=True, transform_input=False).eval()
    fid = fid_score(ddim_video, bdia_ddim_video, inception_model)

    print("DDIM Scores:")
    print(f"Temporal Consistency: {ddim_scores[0]}")
    print(f"Inception Score: {ddim_scores[1]}")
    print(f"Prompt Similarity: {ddim_scores[2]}")

    print("\nBDIA-DDIM Scores:")
    print(f"Temporal Consistency: {bdia_ddim_scores[0]}")
    print(f"Inception Score: {bdia_ddim_scores[1]}")
    print(f"Prompt Similarity: {bdia_ddim_scores[2]}")

    print(f"\nFID Score between DDIM and BDIA-DDIM: {fid}")
