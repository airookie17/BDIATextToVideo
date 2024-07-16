# BDIATextToVideo

This repository contains a project for generating animated videos from text prompts using a custom implementation of the `diffusers` library. The core components of this project include the `AnimateDiffPipeline` and a custom scheduler `BDIADDIMScheduler` added to the `diffusers` folder.

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

- `prompt`: The text prompt to generate the video.
- `negative_prompt`: The negative text prompt to avoid certain features in the video.
- `num_inference_steps`: The number of inference steps.
- `guidance_scale`: The guidance scale.
- `num_frames`: The number of frames in the video.
- `seed`: The random seed for reproducibility.
- `scheduler_type`: The type of scheduler (ddim or bdia-ddim).
- `gamma`: The gamma value for the BDIA-DDIM scheduler (only applicable if scheduler_type is bdia-ddim).
- `output_folder`: The folder where the generated video will be saved.
- `video_name`: The name of the output video file.

## Implementation Details

### AnimateDiffPipeline

The `AnimateDiffPipeline` is used to create the animation pipeline. It leverages the motion adapter and a pretrained model to generate the frames of the video based on the given text prompt.

### Custom BDIADDIMScheduler

The custom `BDIADDIMScheduler` is an enhanced version of the `DDIMScheduler` which includes additional parameters for fine-tuning the denoising process. This scheduler is specifically designed to handle the Back-Door Inference and Adaptive Denoising (BDIA) modifications.

#### Implementation

The BDIADDIMScheduler is implemented in the `scheduling_bdia_ddim.py` file located in the `diffusers` folder. It extends the `DDIMScheduler` and includes additional logic for the BDIA modifications.

Here's a brief overview of the implementation:

1. **Initialization**: The scheduler is initialized with an additional `gamma` parameter, which controls the influence of the BDIA modifications.
2. **Step Function**: The step function is overridden to include the BDIA logic, which adjusts the previous sample based on the current and last denoised samples.
3. **Set Timesteps**: The `set_timesteps` function is also overridden to reset the state for each new denoising process.

