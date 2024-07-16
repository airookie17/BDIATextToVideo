# BDIATextToVideo

This repository contains a project for generating animated videos from text prompts using a custom implementation of the `diffusers` library.

## Contents

- `diffusers/`: Custom diffusers library.
- `videos/`: Folder where generated videos are saved.
- `generate_videos.py`: Script to generate videos using specified parameters.
- `sample.py`: Script to install dependencies and run `generate_videos.py` with default parameters.

## Setup

### Prerequisites

- Python 3.8 or higher
- Git
- An internet connection to download necessary models and dependencies

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/airookie17/BDIATextToVideo.git
   cd BDIATextToVideo
  

2. **Install the required packages**:
  
   ```bash
   pip install torch torchaudio torchvision accelerate
   cd diffusers
   pip install -e .
   cd ..


## Usage

### Running the Sample Script

The sample.py script installs the necessary libraries, sets up the environment, and runs the generate_videos.py script with default parameters.

To run the sample script, use the following command:

   ```bash
   python sample.py
```

## Modifying Parameters

You can modify the parameters in the sample.py script to customize the video generation. Here are the parameters you can change:

- prompt: The text prompt to generate the video.
- negative_prompt: The negative text prompt to avoid certain features in the video.
- num_inference_steps: The number of inference steps.
- guidance_scale: The guidance scale.
- num_frames: The number of frames in the video.
- seed: The random seed for reproducibility.
- scheduler_type: The type of scheduler (ddim or bdia-ddim).
- gamma: The gamma value for the BDIA-DDIM scheduler (only applicable if scheduler_type is bdia-ddim).
- output_folder: The folder where the generated video will be saved.
- video_name: The name of the output video file.

