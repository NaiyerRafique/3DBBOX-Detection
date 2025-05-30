## Overview

This document summarizes the key design decisions in the 3D bounding-box prediction pipeline, including:

- **Architecture**: multi‐modal feature extraction and fusion
- **Loss functions**: combining regression and overlap objectives
- **Metrics**: evaluation protocols distinct from training losses
- **CLI Usage**: how to run training, testing, inference, and visualization

A process‐diagram at the end illustrates the overall data flow.

---

## 1. Architecture Choice

1. **Vision Transformer (ViT) Backbone** 
   - Adopted `vit_b_16` pretrained on ImageNet for strong 2D feature extraction from RGB inputs. 
   - Transformers capture the long distance relationships between different parts of the image. 


2. **Mask & Point-Cloud Streams** 
   - **Mask branch**: a lightweight 2‐layer CNN processes per‐instance for both the Mask and the 	  point cloud.
   - Both use `Conv2d` → ReLU → pooling → `AdaptiveAvgPool2d`

3. **Feature Fusion & Head** 
   - Concatenate ViT features (768-dim for `vit_b_16`) with mask (32) and pc (32) embeddings → 832-dim vector. 
   - **MLP head**: two hidden layers (512 units) with ReLU+dropout, ending in a `Linear(512 → 24)` layer, reshaped to (8 corners × 3 coords).

4. **Model Size Constraint** 
   - Total parameters ≈ 89M, under the 100M limit. 


---

## 2. Loss Functions

- **L1 (Mean Absolute Error)** 
  - Measures per‐corner coordinate differences
  - Encourages accurate localization of each corner.

- **3D IoU Loss** 
  - Based on axis‐aligned bounding volumes computed from corner minima/maxima. 
  - Captures volumetric overlap, penalizing mis‐sized or poorly aligned boxes.

- **Combined Objective** 
  - Balances pointwise accuracy and holistic shape alignment.

---

## 3. Metrics vs. Loss Functions

- **Training losses** (L1 & IoU) serve as differentiable objectives guiding optimization. 
- **Evaluation metrics** should be chosen separately; common choices include:
  - **Mean L1 error** (same as training L1) for direct comparability. 
  - **Average 3D IoU** over the test set, for volumetric accuracy. 
 

> **Note:** Loss functions _inspire_ metrics but metrics can include additional measures (e.g., per‐axis rotation error) to fully characterize model performance.

---

## 4. CLI Usage

Below are the key commands to run each stage of the pipeline.


# How to start
Unzip the folder and place the unzipped dl_challenge dataset next to the scripts in the src folder. This way you dont need to provide /path/to/data in any commands. Then simply run the commands below starting with training.


# Install dependencies
pip install -r requirements.txt

### 4.2. Training

python train.py \
  --data /path/to/data \
  --epochs 30 \
  --bs 16 \
  --lr 1e-4 \
  --alpha 1.0 \
  --out checkpoints \
  --logdir runs \
  --max-models


### 4.3. Validation & Testing

# After training, evaluate on validation/test splits

python test.py \
  --data /path/to/data \
  --ckpt checkpoints/epochxx_xxx.1234.pth \
  --bs 16


### 4.4. PyTorch Inference

python torch_inference.py \
  --ckpt checkpoints/epochxx_xxx.1234.pth \
  --img /path/to/sample/rgb.jpg


### 4.5. ONNX Export & Inference

python inference.py \
  --ckpt checkpoints/epochxx_xxx.1234.pth \
  --img /path/to/sample/rgb.jpg \
  --onnx model.onnx



---

## 5. Process Diagram


flowchart TB
  Inputs:
    RGB[RGB Image]
    MSK[Instance Mask]
    PC[Point Cloud (XYZ)]

  RGB -->|Crop & Resize\224×224| PR1[Preprocess]
  MSK -->|Crop & Resize\224×224| PR2[Preprocess]
  PC  -->|Crop & Resize\224×224| PR3[Preprocess]

  PR1 -->|Tensor\B×3×224×224| F_img[ViT Backbone → 768-d]
  PR2 -->|Tensor\B×1×224×224| F_msk[Mask CNN → 32-d]
  PR3 -->|Tensor\B×3×224×224| F_pc[PC CNN → 32-d]

  F_img & F_msk & F_pc --> Fuse[Concatenate → 832-d]
  Fuse --> MLP[MLP Head → 8×3 coords]
  MLP --> Pred[Predicted 3D Box]

  Training losses:
    Pred -->|vs GT corners| L1[L1 Loss]
    Pred -->|vs GT boxes| IoU[3D IoU Loss]
    L1 & IoU --> Total[Total Loss\n(L1 + α·IoU)]


  Evaluation metrics:
    Pred -->|Compare to GT| M1[Mean L1 Error]
    Pred -->|Compute overlap| M2[Average 3D IoU]



---

