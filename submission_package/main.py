import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision.models import resnet18


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class Cifar10ResNet18(nn.Module):
    """ResNet-18 with CIFAR-10 classifier head and in-model normalization."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 10)

        mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.backbone(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fgsm_attack(image: torch.Tensor, epsilon: float, data_grad: torch.Tensor) -> torch.Tensor:
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, 0.0, 1.0)


def safe_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten()
    b_flat = b.flatten()
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def save_image(img: np.ndarray, title: str, path: Path) -> None:
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Grad-CAM robustness on CIFAR-10 under FGSM perturbations."
    )
    parser.add_argument("--data-root", type=str, default="./data", help="Path for CIFAR-10 data.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/cifar10_resnet18.pth",
        help="Path to CIFAR-10 ResNet-18 checkpoint.",
    )
    parser.add_argument("--num-samples", type=int, default=100, help="Number of random test samples.")
    parser.add_argument(
        "--num-visuals",
        type=int,
        default=10,
        help="How many sampled examples to save as images.",
    )
    parser.add_argument("--epsilon", type=float, default=0.02, help="FGSM perturbation strength.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--save-dir", type=str, default=".", help="Directory for output figures.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Compute device.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    if args.num_visuals < 0:
        raise ValueError("--num-visuals must be >= 0.")
    if not 0.0 <= args.epsilon <= 1.0:
        raise ValueError("--epsilon must be in [0, 1].")

    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Checkpoint not found at "
            f"{checkpoint_path}. Train one first using train_cifar10.py."
        )

    transform = transforms.ToTensor()
    dataset = torchvision.datasets.CIFAR10(
        root=args.data_root,
        train=False,
        download=True,
        transform=transform,
    )

    model = Cifar10ResNet18().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # For 32x32 CIFAR inputs, an earlier stage gives more stable spatial saliency.
    target_layers = [model.backbone.layer3[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    sample_count = min(args.num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), sample_count)

    similarities = []
    attack_successes = 0

    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        input_tensor = image.unsqueeze(0).to(device)
        input_tensor.requires_grad_(True)

        logits = model(input_tensor)
        clean_pred = int(torch.argmax(logits, dim=1).item())

        target = torch.tensor([clean_pred], device=device)
        loss = F.cross_entropy(logits, target)

        model.zero_grad(set_to_none=True)
        loss.backward()
        data_grad = input_tensor.grad.detach()

        adv_tensor = fgsm_attack(input_tensor, args.epsilon, data_grad)

        with torch.no_grad():
            adv_logits = model(adv_tensor)
            adv_pred = int(torch.argmax(adv_logits, dim=1).item())
        if adv_pred != clean_pred:
            attack_successes += 1

        targets = [ClassifierOutputTarget(clean_pred)]
        clean_cam = cam(input_tensor=input_tensor.detach(), targets=targets)[0]
        adv_cam = cam(input_tensor=adv_tensor.detach(), targets=targets)[0]

        similarity = safe_cosine_similarity(clean_cam, adv_cam)
        similarities.append(similarity)

        if i < args.num_visuals:
            clean_img = image.permute(1, 2, 0).cpu().numpy()
            adv_img = adv_tensor.squeeze(0).detach().permute(1, 2, 0).cpu().numpy()

            clean_overlay = show_cam_on_image(clean_img, clean_cam, use_rgb=True)
            adv_overlay = show_cam_on_image(adv_img, adv_cam, use_rgb=True)

            prefix = f"sample_{i:03d}"

            save_image(
                clean_img,
                f"Original (GT: {CIFAR10_CLASSES[label]}, Pred: {CIFAR10_CLASSES[clean_pred]})",
                save_dir / f"{prefix}_original.png",
            )
            save_image(
                adv_img,
                f"Adversarial (Pred: {CIFAR10_CLASSES[adv_pred]})",
                save_dir / f"{prefix}_adversarial.png",
            )
            save_image(clean_overlay, "Grad-CAM (Clean)", save_dir / f"{prefix}_gradcam_clean.png")
            save_image(adv_overlay, "Grad-CAM (Adversarial)", save_dir / f"{prefix}_gradcam_adversarial.png")

    similarities_np = np.array(similarities, dtype=np.float32)
    print("=== Robustness Summary ===")
    print(f"Samples evaluated: {sample_count}")
    print(f"FGSM epsilon: {args.epsilon}")
    print(f"Mean cosine similarity: {similarities_np.mean():.4f}")
    print(f"Std cosine similarity: {similarities_np.std():.4f}")
    print(f"Min cosine similarity: {similarities_np.min():.4f}")
    print(f"Max cosine similarity: {similarities_np.max():.4f}")
    print(f"Attack success rate: {attack_successes / sample_count:.4f}")
    print(f"Saved visual outputs in: {save_dir.resolve()}")


if __name__ == "__main__":
    main()