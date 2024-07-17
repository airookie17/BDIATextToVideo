import subprocess

# Define the parameters
prompts = ["Spiderman is surfing"]
num_frames = 16
num_inference_steps = 40
seed = 0
gamma = 1.0

# Construct the command to run the generate_videos_sd.py script
command = [
    "python", "generate_videos_sd.py",
    "--prompts", *prompts,
    "--num_frames", str(num_frames),
    "--num_inference_steps", str(num_inference_steps),
    "--seed", str(seed),
    "--gamma", str(gamma)
]

# Run the command
subprocess.run(command)
