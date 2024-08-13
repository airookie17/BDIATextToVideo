# Gamma 0.5, 25 inference steps
!python generate_videos_adiff.py --prompt "a cat with glasses reading a book, high quality, bestquality, highlydetailed, ultradetailed" --num_frames 16 --num_inference_steps 25 --guidance_scale 7.5 --scheduler_type bdia-ddim --gamma 0.5 --video_name "bdia_ddim_CAT_0.5_25" --seed 42

# Gamma 0.5, 40 inference steps
!python generate_videos_adiff.py --prompt "a cat with glasses reading a book, high quality, bestquality, highlydetailed, ultradetailed" --num_frames 16 --num_inference_steps 40 --guidance_scale 7.5 --scheduler_type bdia-ddim --gamma 0.5 --video_name "bdia_ddim_CAT_0.5_40" --seed 42

# Gamma 0.2, 25 inference steps
!python generate_videos_adiff.py --prompt "a cat with glasses reading a book, high quality, bestquality, highlydetailed, ultradetailed" --num_frames 16 --num_inference_steps 25 --guidance_scale 7.5 --scheduler_type bdia-ddim --gamma 0.2 --video_name "bdia_ddim_CAT_0.2_25" --seed 42

# Gamma 0.2, 40 inference steps
!python generate_videos_adiff.py --prompt "a cat with glasses reading a book, high quality, bestquality, highlydetailed, ultradetailed" --num_frames 16 --num_inference_steps 40 --guidance_scale 7.5 --scheduler_type bdia-ddim --gamma 0.2 --video_name "bdia_ddim_CAT_0.2_40" --seed 42

# Gamma 1.0, 25 inference steps
!python generate_videos_adiff.py --prompt "a cat with glasses reading a book, high quality, bestquality, highlydetailed, ultradetailed" --num_frames 16 --num_inference_steps 25 --guidance_scale 7.5 --scheduler_type bdia-ddim --gamma 1.0 --video_name "bdia_ddim_CAT_1.0_25" --seed 42

# Gamma 1.0, 40 inference steps
!python generate_videos_adiff.py --prompt "a cat with glasses reading a book, high quality, bestquality, highlydetailed, ultradetailed" --num_frames 16 --num_inference_steps 40 --guidance_scale 7.5 --scheduler_type bdia-ddim --gamma 1.0 --video_name "bdia_ddim_CAT_1.0_40" --seed 42

# DDIM, 25 inference steps
!python generate_videos_adiff.py --prompt "a cat with glasses reading a book, high quality, bestquality, highlydetailed, ultradetailed" --num_frames 16 --num_inference_steps 25 --guidance_scale 7.5 --scheduler_type ddim --video_name "ddim_CAT_25" --seed 42

# DDIM, 40 inference steps
!python generate_videos_adiff.py --prompt "a cat with glasses reading a book, high quality, bestquality, highlydetailed, ultradetailed" --num_frames 16 --num_inference_steps 40 --guidance_scale 7.5 --scheduler_type ddim --video_name "ddim_CAT_40" --seed 42
