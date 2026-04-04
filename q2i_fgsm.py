"""
Q2(i): Train ResNet-18 on CIFAR-10, then compare FGSM from scratch vs IBM ART
Author: Tushar Yadav | Roll No: B23BB1043
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ART imports
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod as ART_FGSM

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
NUM_EPOCHS = 20
LR         = 0.1
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)
NUM_CLASSES  = 10
CLASSES      = ["airplane","automobile","bird","cat","deer",
                "dog","frog","horse","ship","truck"]


# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────
def get_cifar10():
    train_t = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_set = datasets.CIFAR10("./data", train=True,  download=True, transform=train_t)
    test_set  = datasets.CIFAR10("./data", train=False, download=True, transform=test_t)
    train_l   = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    test_l    = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    return train_l, test_l, test_set


# ─────────────────────────────────────────────
# ResNet-18 (pretrained ImageNet → finetune CIFAR-10)
# ─────────────────────────────────────────────
def build_resnet18():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(DEVICE)


# ─────────────────────────────────────────────
# Train / Eval
# ─────────────────────────────────────────────
def train(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss, correct, total = 0., 0, 0
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            loss = criterion(model(imgs), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        correct    += (model(imgs).argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total * 100


@torch.no_grad()
def evaluate_clean(model, loader):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        correct += (model(imgs).argmax(1) == labels).sum().item()
        total   += imgs.size(0)
    return correct / total * 100


# ─────────────────────────────────────────────
# FGSM from scratch
# ─────────────────────────────────────────────
def fgsm_scratch(model, images, labels, epsilon):
    """Return adversarial images using manual FGSM."""
    model.eval()
    images = images.clone().detach().to(DEVICE).requires_grad_(True)
    labels = labels.to(DEVICE)
    loss   = nn.CrossEntropyLoss()(model(images), labels)
    model.zero_grad()
    loss.backward()
    adv = images + epsilon * images.grad.sign()
    # Clamp to valid range (normalised)
    lo = torch.tensor([(0 - m) / s for m, s in zip(CIFAR10_MEAN, CIFAR10_STD)],
                      device=DEVICE).view(3,1,1)
    hi = torch.tensor([(1 - m) / s for m, s in zip(CIFAR10_MEAN, CIFAR10_STD)],
                      device=DEVICE).view(3,1,1)
    return torch.clamp(adv, lo, hi).detach()


@torch.no_grad()
def eval_fgsm_scratch(model, loader, epsilon):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        adv    = fgsm_scratch(model, imgs, labels, epsilon)
        preds  = model(adv).argmax(1)
        correct += (preds == labels.to(DEVICE)).sum().item()
        total   += labels.size(0)
    return correct / total * 100


# ─────────────────────────────────────────────
# FGSM via IBM ART
# ─────────────────────────────────────────────
def build_art_classifier(model):
    criterion = nn.CrossEntropyLoss()
    # ART needs un-normalised [0,1] input; we wrap the normalised model
    # by using mean/std as preprocessing in ART
    mean = np.array(CIFAR10_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std  = np.array(CIFAR10_STD,  dtype=np.float32).reshape(1, 3, 1, 1)
    art_model = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        input_shape=(3, 32, 32),
        nb_classes=NUM_CLASSES,
        preprocessing=(mean, std),
        clip_values=(0.0, 1.0),
        device_type="gpu" if torch.cuda.is_available() else "cpu",
    )
    return art_model


def eval_fgsm_art(art_model, test_set, epsilon, n_samples=1000):
    """Evaluate ART FGSM on first n_samples test images."""
    raw_imgs   = np.array([np.array(test_set[i][0]) / 255.0
                           for i in range(n_samples)], dtype=np.float32)
    # CIFAR images are PIL; need (N,3,32,32)
    raw_imgs   = np.array([np.transpose(np.array(test_set.data[i]), (2,0,1))
                           for i in range(n_samples)], dtype=np.float32) / 255.0
    true_labels = np.array(test_set.targets[:n_samples])

    attack  = ART_FGSM(estimator=art_model, eps=epsilon, eps_step=epsilon,
                        batch_size=64, minimal=False)
    adv_art = attack.generate(x=raw_imgs)

    preds   = np.argmax(art_model.predict(adv_art), axis=1)
    acc     = (preds == true_labels).mean() * 100
    return acc, raw_imgs[:8], adv_art[:8], true_labels[:8]


# ─────────────────────────────────────────────
# Visualise side-by-side
# ─────────────────────────────────────────────
def denorm(img_tensor):
    """Denormalise a (3,H,W) tensor → numpy (H,W,3) in [0,1]."""
    mean = torch.tensor(CIFAR10_MEAN).view(3,1,1)
    std  = torch.tensor(CIFAR10_STD).view(3,1,1)
    img  = img_tensor.cpu() * std + mean
    return img.permute(1,2,0).clamp(0,1).numpy()


def visualise_attacks(model, test_loader, test_set, art_model, epsilon, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    imgs, labels = next(iter(test_loader))
    imgs, labels = imgs[:8], labels[:8]

    adv_scratch = fgsm_scratch(model, imgs, labels, epsilon)

    # ART (raw [0,1] numpy)
    raw_np  = np.array([np.transpose(np.array(test_set.data[i]), (2,0,1))
                         for i in range(8)], dtype=np.float32) / 255.0
    attack  = ART_FGSM(estimator=art_model, eps=epsilon, eps_step=epsilon,
                        batch_size=8, minimal=False)
    adv_art = attack.generate(x=raw_np)          # (8,3,32,32) in [0,1]

    fig, axes = plt.subplots(3, 8, figsize=(20, 8))
    fig.suptitle(f"FGSM ε={epsilon} | Top: Original  Mid: Scratch  Bot: ART", fontsize=12)
    for i in range(8):
        axes[0,i].imshow(denorm(imgs[i]))
        axes[0,i].set_title(CLASSES[labels[i]], fontsize=7)
        axes[0,i].axis("off")

        axes[1,i].imshow(denorm(adv_scratch[i]))
        axes[1,i].axis("off")

        axes[2,i].imshow(np.transpose(adv_art[i], (1,2,0)).clip(0,1))
        axes[2,i].axis("off")

    plt.tight_layout()
    path = os.path.join(save_dir, f"fgsm_comparison_eps{int(epsilon*100)}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    return path


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    wandb.login()
    wandb.init(project="B23BB1043-Assignment5-Q2i", name="ResNet18-FGSM")

    train_loader, test_loader, test_set = get_cifar10()
    model     = build_resnet18()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler    = torch.cuda.amp.GradScaler()

    os.makedirs("checkpoints", exist_ok=True)
    best_acc = 0.0

    # ── Train ──
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0., 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                out  = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * imgs.size(0)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

        tr_loss = total_loss / total
        tr_acc  = correct / total * 100
        val_acc = evaluate_clean(model, test_loader)
        scheduler.step()

        wandb.log({"epoch": epoch, "train_loss": tr_loss,
                   "train_acc": tr_acc, "val_acc": val_acc})
        print(f"Epoch {epoch:2d} | Loss {tr_loss:.4f} | TrainAcc {tr_acc:.2f}% | ValAcc {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/resnet18_cifar10_best.pt")

    print(f"\nBest clean test accuracy: {best_acc:.2f}%")
    assert best_acc >= 72.0, f"Target ≥72% not reached ({best_acc:.2f}%) — train longer!"

    # ── FGSM attacks ──
    model.load_state_dict(torch.load("checkpoints/resnet18_cifar10_best.pt"))
    art_model = build_art_classifier(model)

    epsilons = [0.01, 0.02, 0.03, 0.05, 0.1]
    print("\n{'='*60}")
    print(f"{'ε':>6} | {'Clean':>8} | {'Scratch':>10} | {'ART':>8}")
    print("-"*40)

    wandb_table = wandb.Table(columns=["epsilon","clean_acc","fgsm_scratch_acc","fgsm_art_acc"])
    for eps in epsilons:
        clean_acc   = evaluate_clean(model, test_loader)
        scratch_acc = eval_fgsm_scratch(model, test_loader, eps)
        art_acc, raw_imgs, adv_art_imgs, lab = eval_fgsm_art(art_model, test_set, eps)
        print(f"{eps:>6.3f} | {clean_acc:>7.2f}% | {scratch_acc:>9.2f}% | {art_acc:>7.2f}%")
        wandb_table.add_data(eps, clean_acc, scratch_acc, art_acc)
        wandb.log({
            f"fgsm_scratch_acc_eps{eps}": scratch_acc,
            f"fgsm_art_acc_eps{eps}": art_acc,
        })

    wandb.log({"fgsm_comparison_table": wandb_table})

    # Visualise at ε=0.03
    vis_path = visualise_attacks(model, test_loader, test_set, art_model, epsilon=0.03)
    wandb.log({"fgsm_visual_comparison": wandb.Image(vis_path)})

    # ── WandB: 10 samples clean + adv FGSM (scratch & ART) ──
    imgs_sample, lbl_sample = next(iter(test_loader))
    adv_sample_scratch = fgsm_scratch(model, imgs_sample[:10], lbl_sample[:10], 0.03)
    raw_np = np.array([np.transpose(np.array(test_set.data[i]), (2,0,1))
                       for i in range(10)], dtype=np.float32) / 255.0
    attack = ART_FGSM(estimator=art_model, eps=0.03, eps_step=0.03, batch_size=10)
    adv_art_10 = attack.generate(x=raw_np)

    sample_imgs = []
    for i in range(10):
        sample_imgs.append(wandb.Image(denorm(imgs_sample[i]), caption=f"Clean {CLASSES[lbl_sample[i]]}"))
        sample_imgs.append(wandb.Image(denorm(adv_sample_scratch[i]), caption=f"FGSM-Scratch {CLASSES[lbl_sample[i]]}"))
        sample_imgs.append(wandb.Image(np.transpose(adv_art_10[i],(1,2,0)).clip(0,1), caption=f"FGSM-ART {CLASSES[lbl_sample[i]]}"))
    wandb.log({"fgsm_10_samples": sample_imgs})

    wandb.finish()
    print("\nQ2(i) complete ✓")


if __name__ == "__main__":
    main()
