"""
Q2(ii): Adversarial Detection using ResNet-34 on CIFAR-10
         - Detector A: trained on clean + PGD adversarial examples
         - Detector B: trained on clean + BIM adversarial examples
Author: Tushar Yadav | Roll No: B23BB1043
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms, models
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescent as ART_PGD
from art.attacks.evasion import BasicIterativeMethod as ART_BIM

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE  = 128
DET_EPOCHS  = 15
LR          = 1e-3
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)
NUM_CLASSES  = 10
CLASSES      = ["airplane","automobile","bird","cat","deer",
                "dog","frog","horse","ship","truck"]


# ─────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────
def get_cifar10_raw(n_samples: int = 5000):
    """Return raw [0,1] numpy arrays (N,3,32,32) and labels."""
    test_t   = transforms.Compose([transforms.ToTensor()])
    test_set = datasets.CIFAR10("./data", train=False, download=True, transform=test_t)
    loader   = DataLoader(test_set, batch_size=n_samples, shuffle=False)
    imgs, labels = next(iter(loader))
    return imgs.numpy(), labels.numpy()


def normalise_batch(imgs_np: np.ndarray) -> torch.Tensor:
    """imgs_np: (N,3,32,32) in [0,1] → normalised tensor."""
    mean = torch.tensor(CIFAR10_MEAN).view(1,3,1,1)
    std  = torch.tensor(CIFAR10_STD).view(1,3,1,1)
    t    = torch.tensor(imgs_np, dtype=torch.float32)
    return (t - mean) / std


# ─────────────────────────────────────────────
# Victim ResNet-18 (loaded from Q2-i checkpoint)
# ─────────────────────────────────────────────
def load_victim():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    ckpt = "checkpoints/resnet18_cifar10_best.pt"
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        print(f"Loaded victim from {ckpt}")
    else:
        raise FileNotFoundError(
            f"{ckpt} not found. Run q2i_fgsm.py first to train ResNet-18.")
    model.eval()
    return model.to(DEVICE)


def build_art_victim(model):
    mean = np.array(CIFAR10_MEAN, dtype=np.float32).reshape(1,3,1,1)
    std  = np.array(CIFAR10_STD,  dtype=np.float32).reshape(1,3,1,1)
    return PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        input_shape=(3,32,32),
        nb_classes=NUM_CLASSES,
        preprocessing=(mean, std),
        clip_values=(0.0, 1.0),
        device_type="gpu" if torch.cuda.is_available() else "cpu",
    )


# ─────────────────────────────────────────────
# Generate adversarial examples
# ─────────────────────────────────────────────
def generate_pgd(art_model, clean_np: np.ndarray, eps=0.03) -> np.ndarray:
    print("Generating PGD adversarial examples …")
    attack = ART_PGD(
        estimator=art_model,
        eps=eps, eps_step=eps/4, max_iter=40,
        batch_size=64, verbose=False,
    )
    return attack.generate(x=clean_np)


def generate_bim(art_model, clean_np: np.ndarray, eps=0.03) -> np.ndarray:
    print("Generating BIM adversarial examples …")
    attack = ART_BIM(
        estimator=art_model,
        eps=eps, eps_step=0.01, max_iter=50,
        batch_size=64, verbose=False,
    )
    return attack.generate(x=clean_np)


# ─────────────────────────────────────────────
# Detector: ResNet-34 binary classifier
# ─────────────────────────────────────────────
def build_detector():
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)      # binary: clean=0, adv=1
    return model.to(DEVICE)


def make_detector_dataset(clean_np: np.ndarray, adv_np: np.ndarray):
    """Combine clean (label=0) + adversarial (label=1) into a TensorDataset."""
    clean_t = normalise_batch(clean_np)
    adv_t   = normalise_batch(adv_np)
    X       = torch.cat([clean_t, adv_t], dim=0)
    y       = torch.cat([torch.zeros(len(clean_t), dtype=torch.long),
                         torch.ones( len(adv_t),   dtype=torch.long)], dim=0)
    # shuffle
    perm = torch.randperm(len(X))
    return TensorDataset(X[perm], y[perm])


def train_detector(detector, dataset, run_name: str):
    n_val    = int(len(dataset) * 0.15)
    n_train  = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(42))
    train_l  = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_l    = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(detector.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=DET_EPOCHS)
    scaler    = torch.cuda.amp.GradScaler()

    best_val_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, DET_EPOCHS + 1):
        detector.train()
        total_loss, correct, total = 0., 0, 0
        for X, y in tqdm(train_l, desc=f"[{run_name}] Epoch {epoch}"):
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = detector(X)
                loss   = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * X.size(0)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += X.size(0)

        tr_loss = total_loss / total
        tr_acc  = correct / total * 100

        detector.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for X, y in val_l:
                X, y = X.to(DEVICE), y.to(DEVICE)
                v_correct += (detector(X).argmax(1) == y).sum().item()
                v_total   += y.size(0)
        val_acc = v_correct / v_total * 100
        scheduler.step()

        wandb.log({f"{run_name}/train_loss": tr_loss,
                   f"{run_name}/train_acc":  tr_acc,
                   f"{run_name}/val_acc":    val_acc,
                   "epoch": epoch})
        print(f"Epoch {epoch:2d} | Loss {tr_loss:.4f} | TrainAcc {tr_acc:.2f}% | ValAcc {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(detector.state_dict(), f"checkpoints/detector_{run_name}_best.pt")

    print(f"Best val acc [{run_name}]: {best_val_acc:.2f}%")
    return best_val_acc


# ─────────────────────────────────────────────
# Visualise samples
# ─────────────────────────────────────────────
def log_adv_samples(clean_np, adv_np, labels, attack_name: str, n=10):
    """Log n pairs of (clean, adv) to WandB."""
    imgs_log = []
    for i in range(n):
        c = np.transpose(clean_np[i], (1,2,0)).clip(0,1)
        a = np.transpose(adv_np[i],   (1,2,0)).clip(0,1)
        imgs_log.append(wandb.Image(c, caption=f"Clean  [{CLASSES[labels[i]]}]"))
        imgs_log.append(wandb.Image(a, caption=f"{attack_name} [{CLASSES[labels[i]]}]"))
    wandb.log({f"{attack_name}_samples": imgs_log})


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    wandb.login()
    wandb.init(project="B23BB1043-Assignment5-Q2ii", name="AdversarialDetection")

    # ── Load victim model ──
    victim     = load_victim()
    art_victim = build_art_victim(victim)

    # ── Get clean test images (5000) ──
    clean_np, true_labels = get_cifar10_raw(n_samples=5000)

    # ── Generate attacks ──
    pgd_np = generate_pgd(art_victim, clean_np, eps=0.03)
    bim_np = generate_bim(art_victim, clean_np, eps=0.03)

    # ── Log 10 samples to WandB ──
    log_adv_samples(clean_np, pgd_np, true_labels, "PGD",  n=10)
    log_adv_samples(clean_np, bim_np, true_labels, "BIM",  n=10)

    # ── Detector A: PGD ──
    print("\n" + "="*50)
    print("Training Detector A (PGD)")
    print("="*50)
    det_a   = build_detector()
    ds_pgd  = make_detector_dataset(clean_np, pgd_np)
    acc_pgd = train_detector(det_a, ds_pgd, run_name="PGD_Detector")

    # ── Detector B: BIM ──
    print("\n" + "="*50)
    print("Training Detector B (BIM)")
    print("="*50)
    det_b   = build_detector()
    ds_bim  = make_detector_dataset(clean_np, bim_np)
    acc_bim = train_detector(det_b, ds_bim, run_name="BIM_Detector")

    # ── Summary ──
    print("\n" + "="*50)
    print("Detection Summary")
    print("="*50)
    print(f"PGD Detector Val Accuracy : {acc_pgd:.2f}%")
    print(f"BIM Detector Val Accuracy : {acc_bim:.2f}%")

    wandb.log({"summary/pgd_detection_acc": acc_pgd,
               "summary/bim_detection_acc": acc_bim})

    # ── Bar chart comparison ──
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(["PGD Detector", "BIM Detector"], [acc_pgd, acc_bim], color=["steelblue","tomato"])
    ax.set_ylabel("Detection Accuracy (%)")
    ax.set_title("PGD vs BIM Adversarial Detection")
    ax.axhline(70, linestyle="--", color="black", label="70% target")
    ax.legend()
    ax.set_ylim(0, 100)
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/detection_comparison.png", dpi=120)
    wandb.log({"detection_comparison": wandb.Image("plots/detection_comparison.png")})
    plt.close()

    assert acc_pgd >= 70, f"PGD detector below 70% ({acc_pgd:.2f}%)"
    assert acc_bim >= 70, f"BIM detector below 70% ({acc_bim:.2f}%)"
    print("\nQ2(ii) complete ✓")
    wandb.finish()


if __name__ == "__main__":
    main()
