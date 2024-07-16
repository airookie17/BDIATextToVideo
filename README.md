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
  

2.**Install the required packages**:
  
   ```bash
   pip install torch torchaudio torchvision accelerate
   cd diffusers
   pip install -e .
   cd ..
```

### Usage

