# Baseline Experiments

## Experiment 1 — Linear Probe Baseline

Model: ViT-B (vit_base_patch16_224)  
Pretraining: ImageNet (supervised)  
Backbone: Frozen  
Classifier: Linear head  

Dataset: CIFAR-10  
Image size: 224 × 224  

Training setup:
- Train subset: 5000 images
- Test set: full CIFAR10 test set
- Epochs: 5
- Optimizer: Adam
- Learning rate: 1e-3

Result:

Accuracy: **93.35%**

Notes:
This experiment evaluates representation quality using linear probing with a frozen backbone.