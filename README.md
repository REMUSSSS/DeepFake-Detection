# Cross-Manipulation Deepfake Detection with Vision-Language Models
AI資訊偵測與倫理_許志仲 作業

## Overview

This project implements prompt-tuning on CLIP for cross-manipulation deepfake detection, following the strict protocol of the Assignment-1-DFD.  
- **Training:** Only on Real_youtube + FaceSwap frames  
- **Testing:** Only on NeuralTextures frames  
- **Backbone:** CLIP ViT-B/32 (frozen)
- **Pretrained weights:** https://huggingface.co/openai/clip-vit-base-patch32 
- **Adaptation:** Prompt tuning + linear classifier

## Folder Structure


## Dataset Preparation

1. **Download** the dataset from the official link (see assignment PDF).
2. **Unzip** the archive so you have the following structure:
    ```
    data/
      Real_youtube/
        video1/
          frame1.png
          ...
      FaceSwap/
        video1/
          frame1.png
          ...
      NeuralTextures/
        video1/
          frame1.png
          ...
    ```
3. **Do not** mix frames between folders.  
4. **No** frames from NeuralTextures should be used for training.

## Environment Setup

- Python version: **3.11.7** (recommended)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Start (run_all.sh)
```
python train.py && python eval.py
```

## Training
```
python train.py
```
- Model weights will be saved to checkpoints/promptclip.pt.
## Inference & Evaluation
```
python eval.py
```
- Results (per-frame, per-video, and overall metrics) will be saved to the results/ folder.

## Expected Run-time
- Training: ~60 minutes on a single NVIDIA RTX 4090 (depends on dataset size and hardware)
- Evaluation: ~5 minutes on the same GPU

## Results
- Metrics: AUC, EER, F1, Accuracy, ROC curve
- Per-video and per-frame scores: see results/predictions.csv, results/prediction.csv
