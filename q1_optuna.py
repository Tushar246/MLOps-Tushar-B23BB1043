"""
Q1 – Step 5: Optuna HPO for LoRA hyperparameters on CIFAR-100 / ViT-S
Author: Tushar Yadav | Roll No: B23BB1043
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from transformers import ViTForImageClassification
from peft import LoraConfig, get_peft_model
import optuna
import wandb
import warnings
warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_NAME  = "WinKawaks/vit-small-patch16-224"
NUM_CLASSES = 100
BATCH_SIZE  = 64
NUM_EPOCHS  = 3           # fewer epochs per trial to keep search tractable
LR          = 2e-4
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)


# ── Data ───────────────────────────────────────────────────────────────────────
def get_dataloaders():
    train_t = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    val_t = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    full_train = datasets.CIFAR100(root="./data", train=True, download=True, transform=train_t)
    val_size   = int(len(full_train) * 0.1)
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(full_train, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(42))
    val_set.dataset.transform = val_t

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader


# ── Model factory ──────────────────────────────────────────────────────────────
def build_model(rank: int, alpha: int, dropout: float):
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True
    )
    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["query", "key", "value"],
        bias="none",
        modules_to_save=["classifier"],
    )
    return get_peft_model(model, cfg)


# ── Objective ──────────────────────────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    rank    = trial.suggest_categorical("rank",    [2, 4, 8, 16])
    alpha   = trial.suggest_categorical("alpha",   [2, 4, 8, 16, 32])
    dropout = trial.suggest_float("dropout", 0.0, 0.3, step=0.05)
    lr      = trial.suggest_float("lr", 1e-5, 5e-4, log=True)

    print(f"\nTrial {trial.number}: rank={rank}, alpha={alpha}, dropout={dropout:.2f}, lr={lr:.6f}")

    model        = build_model(rank, alpha, dropout).to(DEVICE)
    criterion    = nn.CrossEntropyLoss()
    optimizer    = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4
    )
    scaler       = torch.cuda.amp.GradScaler()
    train_loader, val_loader = get_dataloaders()

    best_val_acc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(images).logits
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                logits = model(images).logits
                correct += (logits.argmax(1) == labels).sum().item()
                total   += images.size(0)
        val_acc = correct / total * 100
        best_val_acc = max(best_val_acc, val_acc)
        print(f"  Epoch {epoch}: val_acc={val_acc:.2f}%")

        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_acc


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    wandb.login()
    wandb.init(project="B23BB1043-Assignment5-Q1-Optuna", name="optuna_hpo")

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1),
        study_name="vit_lora_hpo",
    )
    study.optimize(objective, n_trials=20, timeout=7200)

    print("\n" + "="*60)
    print("Best trial:")
    t = study.best_trial
    print(f"  Val Accuracy : {t.value:.2f}%")
    print(f"  Params       : {t.params}")

    # Log best params to WandB
    wandb.log({"best_val_acc": t.value, **t.params})

    # ── Re-train best config for full epochs ──
    from q1_train import (get_dataloaders as get_full_loaders,
                           build_vit_lora, train_experiment)
    from torchvision import datasets

    best_p  = t.params
    train_l, val_l, test_l = get_full_loaders()
    model   = build_vit_lora(rank=best_p["rank"], alpha=best_p["alpha"], dropout=best_p["dropout"])

    res = train_experiment(
        model, train_l, val_l, test_l,
        run_name=f"Optuna_Best_r{best_p['rank']}_a{best_p['alpha']}",
        use_lora=True,
        rank=best_p["rank"],
        alpha=best_p["alpha"],
        dropout=best_p["dropout"],
        save_dir="checkpoints",
    )
    print(f"\nBest model test accuracy: {res['overall_test_acc']:.2f}%")
    wandb.finish()


if __name__ == "__main__":
    main()
