# realtime_segment_detection_jetson
Realtime Language/Text-based Segment &amp; Detection using Jetson Orin AGX


# Realtime Language Segment Anything

A real-time language-guided image and video segmentation application using SAM (Segment Anything Model) and YOLO-World, optimized for NVIDIA Jetson Orin AGX on CUDA.

## Features

- **Image Segmentation**: Upload images and segment objects using natural language prompts
- **Video Segmentation**: Process videos with language-guided segmentation
- **Real-time Webcam**: Live webcam feed with real-time segmentation
- **Web Interface**: Easy-to-use Gradio interface accessible via browser

## Hardwares used for the Real-Time Inference + Images + Videos Inference

- **NVIDIA Jetson Orin AGX**
- Minimum 8GB RAM
- USB Camera or webcam (for real-time features)
- MicroSD card (64GB or larger recommended)

## Software Requirements

PyTorch v2.3.0
JetPack 6.0 (L4T R36.2 / R36.3) + CUDA 12.2
```
torch 2.3 - torch-2.3.0-cp310-cp310-linux_aarch64.whl
torchaudio 2.3 - torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl
torchvision 0.18 - torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl
```
Sidenotes: pip installing torch & torchvision is not recommended. Instead, download through the whls in [here]( https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048).

## Installation

### Requirements
```
pip install -r requirements.txt

```

### Jetson Orin AGX Preparation

In the Jetson Orin AGX, the torchvision, torch version had problems on Jetson Pack 5 which I couldn't find the issue.
Therefore, I followed the JetPack 6.0 Torchv2.3.0 under this [link]( https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) (It is a same link above in software requirements for just who scrolled quick!). 

#### Verify CUDA Installation
```
bash
sudo apt update
sudo apt install -y python3-pip python3-dev python3-venv
sudo apt install -y libopencv-dev python3-opencv
sudo apt install -y libfreetype6-dev libpng-dev libjpeg-dev
sudo apt install -y git wget curl
```


## Project Structure
```
Realtime_Language_Segment_Anything/
├── app.py # Main Gradio application
├── llm.py # EfficientViT-SAM model implementation
├── webcam.py # Real-time processing and webcam handling
├── models/ # Downloaded model files
│ ├── yolov8x-world.pt
│ └── efficientvit_sam_.pt
├── assets/ # Demo images and examples
│ ├── fig/
│ └── demo/
├── requirements.txt # Python dependencies
└── README.md # This file
```
## Run the applicaton
```
python app.py

Check if port 7860 is available
netstat -tulpn | grep 7860
```

'''
## Weights 
Weights can be found in the [Huggingface repo](https://huggingface.co/mit-han-lab/efficientvit-sam/tree/main).
Place the models under `assets/checkpoints/sam` for EfficientViT-SAMs.
YOLO models are automatically downloaded from ultralytics library.

Codes are heaviliy borrowed from [Husky-AI9](https://github.com/Husky-AI9/Realtime_Language_Segment_Anything])
