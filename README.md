# RVOS: Referring Video Object Segmentation

## Overview
This repository contains the implementation of **RVOS**, a project on **Referring Video Object Segmentation (RVOS)**. RVOS aims to segment object instances in a video based on natural language expressions across all frames. The project leverages the **DAVIS-2017** dataset and integrates a custom deep learning model that combines visual and textual features to achieve segmentation and tracking. The implementation includes data preprocessing, model training, validation, and visualization of results.

The project was developed as part of the **EE673 DLCV** course under the supervision of **Santosh Kumar Vipparthi** by **Shiv Prakash (2021EEB1030)** and **Varun Sharma (2021EEB1032)**, submitted on **12/05/2025**.

## Project Description
RVOS is a challenging computer vision task that requires understanding both visual and linguistic inputs to segment objects in videos based on textual descriptions (e.g., "the red car on the left"). Unlike traditional video object segmentation, RVOS does not rely on ground-truth masks in the first frame but instead interprets natural language to identify and track objects across frames.

### Key Features
- **Dataset**: Utilizes the DAVIS-2017 dataset with video sequences, ground-truth segmentation masks, and textual expressions.
- **Model**: A custom RVOS model with:
  - **Visual Backbone**: ResNet50 or ResNet18 (pre-trained on ImageNet).
  - **Text Encoder**: BERT-based encoder for textual expressions.
  - **Cross-Modal Fusion**: Multi-head attention to align visual and textual features.
  - **Temporal Module**: 3D convolutional network for temporal consistency.
  - **Segmentation Head**: Generates per-frame segmentation masks.
- **Training Pipeline**: Includes loss functions (BCE, Dice, Focal), AdamW optimizer, cosine annealing scheduler, mixed precision training, gradient accumulation, and clipping.
- **Visualization**: Displays predicted segmentation masks overlaid on video frames with textual context.

## Requirements
To run the project, ensure the following dependencies are installed:
- Python 3.8+
- PyTorch 1.9+ (with CUDA support for GPU acceleration)
- Transformers (Hugging Face)
- NumPy
- Matplotlib
- Albumentations (for data augmentation)
- tqdm (for progress bars)

## Dataset
The project uses the **DAVIS-2017** dataset, pre-split into training and validation sets. The validation set includes:
- Video frames
- Ground-truth segmentation masks
- Textual expressions describing target objects
- Tokenized text inputs (via BERT tokenizer)
- Video IDs

Preprocessing steps include:
- Normalizing frames (mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225])
- Resizing masks to match model output dimensions
- Tokenizing text inputs using BERT's `bert-base-uncased` tokenizer

## Model Architecture
The RVOS model (`RVOSModel`) consists of:
1. **Visual Backbone**: Extracts features from video frames using ResNet50 or ResNet18.
2. **Text Encoder**: Encodes textual expressions using BERT (`bert-base-uncased`).
3. **Cross-Modal Fusion**: Aligns visual and textual features via multi-head attention.
4. **Temporal Module**: Captures temporal dependencies using 3x3x3 convolutions.
5. **Segmentation Head**: Produces per-frame segmentation masks with bilinear upsampling.

### Training Details
- **Loss Function**: Combines Binary Cross-Entropy (weight: 0.3), Dice Loss (weight: 0.6), and Focal Loss (weight: 0.1).
- **Optimizer**: AdamW (learning rate: 5e-4, weight decay: 1e-4).
- **Scheduler**: Cosine Annealing with Warm Restarts (T_0=5, T_mult=2).
- **Mixed Precision**: Enabled via PyTorch AMP for efficiency.
- **Gradient Accumulation**: Accumulates gradients over 4 steps.
- **Gradient Clipping**: Max norm of 1.0.

## Results
### Training
- **IoU**: Improved from 0.4720 (epoch 1) to 0.6207 (epoch 50).
- **Loss**: Decreased from 0.3807 to 0.2000.
- **Gradient Norms**: Stable, ranging from 0.4318 to 0.1743.

### Validation
- **Best IoU**: 0.2312 at epoch 17.
- **Validation Loss**: Varied between 0.5165 and 0.6104.
- **Observation**: Significant gap between training and validation IoU suggests overfitting.

### Visualization
- Successfully visualized predicted masks with semi-transparent overlays, confirming correct data loading and model predictions.

## Limitations
- **Generalization**: Struggles with complex scenes or ambiguous text descriptions.
- **Temporal Consistency**: May fail with fast-moving objects or occlusions.
- **Text Understanding**: Limited by complex or context-dependent expressions.
- **Computational Cost**: High GPU memory usage, unsuitable for edge devices.
- **Dataset Size**: Smaller dataset used due to resource constraints.

## Acknowledgments
- **DAVIS-2017** dataset for providing the data.
- **Santosh Kumar Vipparthi** for guidance.
- **PyTorch** and **Hugging Face Transformers** for model development.
