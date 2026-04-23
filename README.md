# Robustness of XAI Explanations under Adversarial Noise

This project evaluates how Grad-CAM explanations change under FGSM adversarial perturbations on CIFAR-10.

## Project Files
- main.py: robustness evaluation and visualization pipeline
- train_cifar10.py: CIFAR-10 ResNet-18 training script
- implementation.md: requirements and implementation report
- requirements.txt: Python dependencies

## Quick Start (Windows PowerShell)
1. Create environment (Python 3.10 or 3.11 recommended):
- "C:\Program Files\Python310\python.exe" -m venv .venv310

2. Install dependencies:
- .\.venv310\Scripts\python.exe -m pip install --upgrade pip
- .\.venv310\Scripts\python.exe -m pip install -r requirements.txt

3. Train checkpoint:
- .\.venv310\Scripts\python.exe .\train_cifar10.py --epochs 10 --batch-size 128 --save-path .\checkpoints\cifar10_resnet18.pth --device cpu --num-workers 0

4. Run robustness evaluation:
- .\.venv310\Scripts\python.exe .\main.py --checkpoint .\checkpoints\cifar10_resnet18.pth --num-samples 100 --epsilon 0.02 --device cpu --save-dir .\outputs

## Outputs
Evaluation saves these files in outputs/:
- original_image.png
- adversarial_image.png
- gradcam_clean.png
- gradcam_adversarial.png

Console output includes:
- Mean/Std/Min/Max Grad-CAM cosine similarity
- Attack success rate

## Notes
- The evaluation targets an earlier convolution block for Grad-CAM to avoid degenerate saliency maps on 32x32 CIFAR inputs.
- For stronger experiments, run multiple seeds and report confidence intervals.

## Multi-Seed Reproducibility (Recommended)
Run additional seeds and keep outputs separated by folder:

Seed 7:
- .\\.venv310\\Scripts\\python.exe .\\train_cifar10.py --epochs 10 --batch-size 128 --save-path .\\checkpoints\\cifar10_resnet18_seed7.pth --device cpu --num-workers 0 --seed 7
- .\\.venv310\\Scripts\\python.exe .\\main.py --checkpoint .\\checkpoints\\cifar10_resnet18_seed7.pth --num-samples 100 --epsilon 0.02 --device cpu --save-dir .\\outputs_seed7 --seed 7

Seed 123:
- .\\.venv310\\Scripts\\python.exe .\\train_cifar10.py --epochs 10 --batch-size 128 --save-path .\\checkpoints\\cifar10_resnet18_seed123.pth --device cpu --num-workers 0 --seed 123
- .\\.venv310\\Scripts\\python.exe .\\main.py --checkpoint .\\checkpoints\\cifar10_resnet18_seed123.pth --num-samples 100 --epsilon 0.02 --device cpu --save-dir .\\outputs_seed123 --seed 123
