# Mobile-Former: Bridging MobileNet and Transformer

## Description

This project implements and trains Mobile-Former models, which combine the efficiency of MobileNets with the expressiveness of Transformers. The work is based on the paper "Mobile-Former: Bridging MobileNet and Transformer" (CVPR 2022).

## Models

The project includes several variants of Mobile-Former models:

- mobile_former_508m
- mobile_former_294m
- mobile_former_214m
- mobile_former_151m
- mobile_former_96m
- mobile_former_52m
- mobile_former_26m

These models vary in size and complexity, offering different trade-offs between performance and computational efficiency.

## Dataset

The models are trained on the ImageNet dataset, which consists of 1000 classes. The training script is set up to work with a restructured and resized version of ImageNet.

## Training

Training is performed using PyTorch. Key training parameters include:

- Optimizer: AdamW
- Learning rate: 1e-3
- Minimum learning rate: 1e-5
- Scheduler: Cosine annealing
- Warmup epochs: 10000
- Weight decay: 0.05
- Gradient clipping: 1.0
- Batch size: 20
- Total epochs: 300

Data augmentation techniques used:
- AutoAugment (rand-m15-n2)
- Mixup (alpha = 0.8)
- Label smoothing (0.1)

## Evaluation

The models are evaluated on the ImageNet validation set. The main metrics used are:

- Top-1 accuracy
- Top-5 accuracy

## Usage

To train a model:

```python
python train.py --model mobile_former_294m --data-dir /path/to/imagenet --batch-size 20 --epochs 300
```

To resume training from a checkpoint:

```python
python train.py --model mobile_former_294m --data-dir /path/to/imagenet --resume /path/to/checkpoint.pth.tar
```

## Requirements

- PyTorch
- torchvision
- timm
- NVIDIA APEX (optional, for mixed precision training)

Note: Specific version requirements should be added based on the actual dependencies used in the project.
