# Baseline Experiments

## Linear Probe Baseline

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

Results:

| Seed | Accuracy |
|-----|------|
| 0 | 93.35 |
| 1 | 93.45 |
| 2 | 93.51 |

Final Result:

**93.44 ± 0.07**



## PathMNIST Baseline (Preliminary)

**Dataset**: PathMNIST (9 tissue types)
**Image size**: 224 × 224
**Training setup**:
- Full training set (~100k images)
- Epochs: 5  
- Optimizer: Adam
- Learning rate: 1e-3

**Result**:

| Seed | Accuracy |
|------|----------|
| 2    | 88.73    |

**Current**: 88.73%  
*Note: Single seed run. More seeds needed for variance estimation.*