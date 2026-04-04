"""
Q1: ViT-S Fine-tuning on CIFAR-100 (with and without LoRA)
Author: Tushar Yadav | Roll No: B23BB1043
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor
from peft import LoraConfig, get_peft_model, TaskType
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MODEL_NAME = "WinKawaks/vit-small-patch16-224"
NUM_CLASSES = 100
BATCH_SIZE = 64
NUM_EPOCHS = 5
LR = 2e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.1
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)


# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────
def get_dataloaders():
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(IMG_SIZE, padding=4),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    full_train = datasets.CIFAR100(root="./data", train=True, download=True, transform=train_transform)
    test_set   = datasets.CIFAR100(root="./data", train=False, download=True, transform=test_transform)

    val_size   = int(len(full_train) * VAL_SPLIT)
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(full_train, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(42))
    # val uses test_transform
    val_set.dataset.transform = test_transform

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
# Model builders
# ─────────────────────────────────────────────
def build_vit_no_lora():
    """Freeze everything except the classification head."""
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[No-LoRA] Trainable parameters: {trainable:,}")
    return model


def build_vit_lora(rank: int, alpha: int, dropout: float = 0.1):
    """Apply LoRA to Q, K, V projections + trainable classification head."""
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )

    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["query", "key", "value"],
        bias="none",
        modules_to_save=["classifier"],   # classifier remains fully trainable
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


# ─────────────────────────────────────────────
# Train / Eval helpers
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, scaler, epoch, run_name):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"[{run_name}] Train Epoch {epoch}")

    # Collect LoRA gradient norms for logging
    lora_grad_norms = []

    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)
            logits  = outputs.logits if hasattr(outputs, "logits") else outputs
            loss    = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Capture LoRA weight grad norms
        for name, param in model.named_parameters():
            if "lora_" in name and param.grad is not None:
                lora_grad_norms.append(param.grad.norm().item())

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / total
    accuracy = correct / total * 100
    avg_lora_grad = np.mean(lora_grad_norms) if lora_grad_norms else 0.0
    return avg_loss, accuracy, avg_lora_grad


@torch.no_grad()
def evaluate(model, loader, criterion, device=DEVICE):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        logits  = outputs.logits if hasattr(outputs, "logits") else outputs
        loss    = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total * 100
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


# ─────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────
def plot_class_accuracy_histogram(preds, labels, run_name, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    per_class_acc = []
    for c in range(NUM_CLASSES):
        mask = labels == c
        if mask.sum() == 0:
            per_class_acc.append(0.0)
        else:
            per_class_acc.append((preds[mask] == c).mean() * 100)

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.bar(range(NUM_CLASSES), per_class_acc)
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Class-wise Test Accuracy — {run_name}")
    plt.tight_layout()
    path = os.path.join(save_dir, f"{run_name}_classwise_acc.png")
    plt.savefig(path, dpi=100)
    plt.close()
    return path


# ─────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────
def train_experiment(model, train_loader, val_loader, test_loader,
                     run_name, use_lora, rank=None, alpha=None, dropout=None,
                     save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)

    wandb.init(
        project="B23BB1043-Assignment5-Q1",
        name=run_name,
        config={
            "model": MODEL_NAME,
            "use_lora": use_lora,
            "rank": rank,
            "alpha": alpha,
            "dropout": dropout,
            "epochs": NUM_EPOCHS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
        },
        reinit=True,
    )

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler    = torch.cuda.amp.GradScaler()

    best_val_acc = 0.0
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_acc, lora_grad = train_one_epoch(model, train_loader, optimizer, criterion, scaler, epoch, run_name)
        val_loss, val_acc, _, _    = evaluate(model, val_loader, criterion)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": tr_loss, "val_loss": val_loss,
            "train_acc": tr_acc,   "val_acc": val_acc,
            "lora_grad_norm": lora_grad,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)

        wandb.log(row)
        print(f"Epoch {epoch:2d} | TR Loss {tr_loss:.4f} Acc {tr_acc:.2f}% | "
              f"VAL Loss {val_loss:.4f} Acc {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(save_dir, f"{run_name}_best.pt")
            if use_lora:
                model.save_pretrained(ckpt_path)
            else:
                torch.save(model.state_dict(), ckpt_path)

    # ── Test ──
    _, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion)
    print(f"\n[{run_name}] Test Accuracy: {test_acc:.2f}%")
    wandb.log({"test_accuracy": test_acc})

    # Class-wise histogram
    hist_path = plot_class_accuracy_histogram(test_preds, test_labels, run_name)
    wandb.log({"class_accuracy_histogram": wandb.Image(hist_path)})

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result = {
        "run_name":          run_name,
        "use_lora":          use_lora,
        "rank":              rank,
        "alpha":             alpha,
        "dropout":           dropout,
        "overall_test_acc":  test_acc,
        "trainable_params":  trainable_params,
        "history":           history,
    }

    with open(os.path.join(save_dir, f"{run_name}_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    wandb.finish()
    return result


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def main(args):
    wandb.login()
    train_loader, val_loader, test_loader = get_dataloaders()
    all_results = []

    # ── Experiment 0: No LoRA baseline ──
    if not args.lora_only:
        model = build_vit_no_lora()
        res   = train_experiment(model, train_loader, val_loader, test_loader,
                                 run_name="NoLoRA",
                                 use_lora=False)
        all_results.append(res)

    # ── Experiments: LoRA combinations ──
    ranks    = [2, 4, 8]
    alphas   = [2, 4, 8]
    dropout  = 0.1
    exp_no   = 1

    for r in ranks:
        for a in alphas:
            run_name = f"LoRA_r{r}_a{a}_d{int(dropout*10)}"
            print(f"\n{'='*60}")
            print(f"Experiment {exp_no}: rank={r}, alpha={a}, dropout={dropout}")
            print(f"{'='*60}")
            model = build_vit_lora(rank=r, alpha=a, dropout=dropout)
            res   = train_experiment(model, train_loader, val_loader, test_loader,
                                     run_name=run_name,
                                     use_lora=True,
                                     rank=r, alpha=a, dropout=dropout)
            res["exp_no"] = exp_no
            all_results.append(res)
            exp_no += 1

    # ── Summary table ──
    print("\n\n" + "="*80)
    print(f"{'Run':<25} {'LoRA':>6} {'Rank':>5} {'Alpha':>6} {'Drop':>5} {'TestAcc':>9} {'Params':>12}")
    print("="*80)
    for r in all_results:
        print(f"{r['run_name']:<25} {str(r['use_lora']):>6} {str(r['rank']):>5} "
              f"{str(r['alpha']):>6} {str(r['dropout']):>5} "
              f"{r['overall_test_acc']:>8.2f}% {r['trainable_params']:>12,}")

    with open("all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_only", action="store_true",
                        help="Skip the no-LoRA baseline")
    args = parser.parse_args()
    main(args)
