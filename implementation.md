# Software Requirements and Implementation Report

## Project Title
Robustness of XAI Explanations under Adversarial Noise

## 1. Objective
This project evaluates how stable Grad-CAM explanations are when CIFAR-10 inputs are perturbed with FGSM adversarial noise. The goal is to quantify explanation drift and provide visual evidence on clean versus adversarial samples.

## 2. Scope
The project includes:
- CIFAR-10 classification with a CIFAR-10-trained ResNet-18
- Grad-CAM explanation generation
- FGSM perturbation generation
- Quantitative comparison of explanation maps using cosine similarity
- Aggregate robustness reporting over multiple random test samples

Out of scope:
- Certified robustness guarantees
- Defenses (adversarial training, input purification)
- Cross-dataset generalization studies

## 3. Functional Requirements
1. Data loading
- Load CIFAR-10 test set
- Randomly sample N images reproducibly (seeded)

2. Model prediction
- Use ResNet-18 with 10-way CIFAR-10 output head
- Load weights from a local checkpoint

3. Explanation generation
- Produce Grad-CAM maps for clean inputs
- Produce Grad-CAM maps for adversarial inputs, using the clean predicted class as target

4. Adversarial attack
- Generate FGSM adversarial inputs with configurable epsilon
- Keep perturbed pixels in valid range [0, 1]

5. Robustness evaluation
- Compute cosine similarity between clean and adversarial Grad-CAM maps
- Report mean, standard deviation, min, and max similarity across samples
- Report attack success rate (prediction flip ratio)

6. Output artifacts
- Save one example set:
  - original_image.png
  - adversarial_image.png
  - gradcam_clean.png
  - gradcam_adversarial.png
- Print summary statistics to console

## 4. Non-Functional Requirements
- Reproducibility: deterministic seed control for NumPy, Python, and PyTorch
- Portability: runs on CPU or CUDA with standard Python dependencies
- Reliability: safe cosine similarity handling when norm is zero
- Usability: command-line arguments for all key settings

## 5. Architecture
Workflow:
1. Load CIFAR-10 image(s)
2. Predict class with CIFAR-10 ResNet-18
3. Generate Grad-CAM for clean image
4. Compute FGSM perturbation
5. Generate Grad-CAM for adversarial image
6. Compare maps with cosine similarity
7. Aggregate and report metrics

## 6. Implementation Summary
Main evaluation script: main.py
- Uses checkpointed CIFAR-10 ResNet-18 model
- Evaluates robustness over multiple random test samples
- Produces both qualitative (images) and quantitative outputs

Training script: train_cifar10.py
- Trains ResNet-18 on CIFAR-10 from scratch
- Saves checkpoint for the evaluation pipeline

Dependencies: requirements.txt
- Torch, torchvision, NumPy, matplotlib, grad-cam

## 7. Why This Version Is Stronger
Compared with an early single-image prototype, this implementation improves:
- Scientific validity: dataset and model are aligned (CIFAR-10 with CIFAR-10 model)
- Evidence quality: reports distribution-level metrics, not one sample only
- Reproducibility: fixed seeds and explicit setup requirements
- Submission readiness: training and evaluation scripts are separated and documented

## 8. How To Run
Prerequisite:
- Use Python 3.10 to 3.12 (recommended: 3.11) for stable PyTorch wheel support.

1. Install dependencies:
- pip install -r requirements.txt

2. Train model (or provide your own checkpoint):
- python train_cifar10.py --epochs 10 --save-path ./checkpoints/cifar10_resnet18.pth

3. Run robustness evaluation:
- python main.py --checkpoint ./checkpoints/cifar10_resnet18.pth --num-samples 100 --epsilon 0.02

Optional flags:
- --device cpu
- --seed 42
- --save-dir ./outputs

## 9. Expected Deliverables for Submission
- Source files:
  - main.py
  - train_cifar10.py
  - requirements.txt
  - implementation.md
- Generated outputs:
  - outputs/original_image.png
  - outputs/adversarial_image.png
  - outputs/gradcam_clean.png
  - outputs/gradcam_adversarial.png
- Console summary from main.py with similarity and attack success metrics

## 10. Experimental Results (Run on April 18, 2026)
Protocol:
- 3 independent seeds: 42, 7, 123
- Training: 10 epochs, batch size 128, CPU
- Evaluation: 100 CIFAR-10 test samples, FGSM epsilon 0.02

Per-seed summary:

| Seed | Checkpoint | Test Accuracy | Mean Cosine Similarity | Similarity Std | Attack Success Rate |
|---|---|---:|---:|---:|---:|
| 42 | checkpoints/cifar10_resnet18.pth | 0.7788 | 0.7324 | 0.3638 | 0.8800 |
| 7 | checkpoints/cifar10_resnet18_seed7.pth | 0.7905 | 0.6801 | 0.3904 | 0.9000 |
| 123 | checkpoints/cifar10_resnet18_seed123.pth | 0.7862 | 0.7820 | 0.2469 | 0.9100 |

Cross-seed aggregate (mean +/- sample std):
- Test accuracy: 0.7852 +/- 0.0059
- Mean cosine similarity: 0.7315 +/- 0.0510
- Attack success rate: 0.8967 +/- 0.0153

Generated visualization folders:
- outputs/
- outputs_seed7/
- outputs_seed123/

Interpretation:
- FGSM changes predicted labels frequently (about 89.7% attack success on average).
- Grad-CAM stability is moderate on average (about 0.73), but variance is high across samples and seeds.
- This indicates explanation robustness is context-dependent, not uniformly stable or uniformly fragile.

## 11. Limitations and Honest Claims
- Results depend on checkpoint quality; poor training reduces analysis reliability
- FGSM is a first-order attack; stronger attacks (PGD/CW) may produce different behavior
- Grad-CAM robustness is one aspect of XAI reliability; not a complete trust guarantee

Conclusion to report:
This project can show whether Grad-CAM explanations are stable or unstable under FGSM on CIFAR-10, based on aggregate statistics, instead of making a universal claim from one image.

## 12. Final Submission Checklist
- Code runs from a clean environment using requirements.txt
- All source files are present: main.py, train_cifar10.py, requirements.txt, README.md, implementation.md
- Trained checkpoints are present in checkpoints/
- Visual outputs are present in outputs/, outputs_seed7/, outputs_seed123/
- Report includes multi-seed results table and aggregate statistics
- Claims in conclusion match measured evidence (no overclaiming)

## 13. Final Abstract (Ready to Use)
This project studies the robustness of Grad-CAM explanations under adversarial perturbations on CIFAR-10. A ResNet-18 model was trained for CIFAR-10 classification and then evaluated with FGSM attacks at epsilon 0.02. For each sample, Grad-CAM heatmaps were generated for clean and adversarial inputs, and explanation drift was quantified using cosine similarity. Across three random seeds, the model achieved mean test accuracy of 0.7852 +/- 0.0059, while adversarial attacks achieved a mean success rate of 0.8967 +/- 0.0153. Explanation similarity averaged 0.7315 +/- 0.0510, with substantial variance across samples, indicating that Grad-CAM stability is neither consistently preserved nor consistently broken under attack. These results show that XAI robustness should be evaluated statistically rather than inferred from single-example visualizations.
