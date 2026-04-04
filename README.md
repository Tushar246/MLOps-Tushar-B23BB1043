# Assignment 5 – ViT Fine-tuning with LoRA & Adversarial Attacks

**Name:** Tushar Yadav  
**Roll No:** B23BB1043  
**Institute:** IIT Jodhpur

---

## 🔗 Important Links

| Resource | Link |
|----------|------|
| 📊 WandB Project — Q1 | [B23BB1043-Assignment5-Q1](https://wandb.ai/b23bb1043-indian-institute-of-technology-jodhpur/B23BB1043-Assignment5-Q1?nw=nwuserb23bb1043) |
| 📊 WandB Project — Q1 Optuna | [B23BB1043-Assignment5-Q1-Optuna](https://wandb.ai/b23bb1043-indian-institute-of-technology-jodhpur/B23BB1043-Assignment5-Q1-Optuna) |
| 📊 WandB Project — Q2 | [B23BB1043-Assignment5-Q2ii](https://wandb.ai/b23bb1043-indian-institute-of-technology-jodhpur/B23BB1043-Assignment5-Q2ii/runs/thpiep6m?nw=nwuserb23bb1043) |
| 🤗 HuggingFace Model | [Tushar04913/vit-cifar100-lora](https://huggingface.co/Tushar04913/vit-cifar100-lora/tree/main) |

---

## 📁 Repository Structure

```
Assignment-5/
├── q1_train.py            # Q1: ViT-S fine-tuning — no-LoRA + all 9 LoRA grid combinations
├── q1_optuna.py           # Q1: Optuna HPO for LoRA hyperparameters
├── q1_train_best.py       # Q1: Re-train best Optuna config (r=4, a=4, d=0.3)
├── q1_upload_hf.py        # Q1: Push best model weights to HuggingFace Hub
├── q2i_fgsm.py            # Q2(i): ResNet-18 training + FGSM scratch vs IBM ART
├── q2ii_detection.py      # Q2(ii): Adversarial Detection with ResNet-34 (PGD + BIM)
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚙️ Installation

### Option 1 — Docker (required by assignment)

```bash
docker build -t ass5 .
docker run --gpus all -it ass5 bash
```

### Option 2 — pip (Google Colab)

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Q1 — ViT-S Fine-tuning on CIFAR-100

```bash
# Step 1: Run no-LoRA baseline + all 9 LoRA grid experiments
python q1_train.py

# Step 2: Optuna hyperparameter search (20 trials)
python q1_optuna.py

# Step 3: Re-train best config (rank=4, alpha=4, dropout=0.3)
python q1_train_best.py

# Step 4: Upload best model to HuggingFace
python q1_upload_hf.py \
  --model_path checkpoints/Optuna_Best_r4_a4_d3_best \
  --run_name   Optuna_Best_r4_a4_d3 \
  --hf_repo    Tushar04913/vit-cifar100-lora
```

### Q2 — Adversarial Attacks

```bash
# Step 1: Train ResNet-18 + FGSM attack comparison (run FIRST)
python q2i_fgsm.py

# Step 2: Train adversarial detectors with ResNet-34 (needs Q2i checkpoint)
python q2ii_detection.py
```

---

## 📊 Q1 Results

### No-LoRA Baseline

> **Test Accuracy: 80.46%** | Trainable Parameters: 38,500

---

### Experiment 1 — Rank: 2, Alpha: 2, Dropout: 0.1

| Epoch | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy |
|-------|--------------|----------------|------------------|-------------------|
| 1  | 1.1800 | 0.4091 | 73.56% | 87.90% |
| 2  | 0.7700 | 0.3967 | 81.47% | 88.78% |
| 3  | 0.3400 | 0.3577 | 89.37% | 89.18% |
| 4  | 0.3100 | 0.3499 | 90.37% | 89.39% |
| 5  | 0.2800 | 0.3414 | 91.33% | 89.76% |
| 6  | 0.2600 | 0.3319 | 91.69% | 89.81% |
| 7  | 0.2400 | 0.3332 | 92.58% | 89.86% |
| 8  | 0.2200 | 0.3309 | 92.91% | 89.91% |
| 9  | 0.2200 | 0.3285 | 93.21% | 90.10% |
| 10 | 0.2200 | 0.3210 | 93.43% | 90.12% |

---

### Experiment 2 — Rank: 2, Alpha: 4, Dropout: 0.1

| Epoch | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy |
|-------|--------------|----------------|------------------|-------------------|
| 1  | 1.1512 | 0.4090 | 73.96% | 87.56% |
| 2  | 1.1512 | 0.4090 | 73.96% | 87.56% |
| 3  | 0.3446 | 0.3652 | 89.55% | 88.50% |
| 4  | 0.3446 | 0.3652 | 89.55% | 88.50% |
| 5  | 0.2740 | 0.3473 | 91.49% | 89.14% |
| 6  | 0.2740 | 0.3473 | 91.49% | 89.14% |
| 7  | 0.2337 | 0.3440 | 92.81% | 89.50% |
| 8  | 0.2337 | 0.3440 | 92.81% | 89.50% |
| 9  | 0.2121 | 0.3402 | 93.61% | 89.56% |
| 10 | 0.2121 | 0.3402 | 93.61% | 89.56% |

---

### Experiment 3 — Rank: 2, Alpha: 8, Dropout: 0.1

| Epoch | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy |
|-------|--------------|----------------|------------------|-------------------|
| 1  | 1.1350 | 0.4160 | 74.22% | 87.64% |
| 2  | 1.1350 | 0.4160 | 74.22% | 87.64% |
| 3  | 0.3392 | 0.3572 | 89.54% | 89.12% |
| 4  | 0.3392 | 0.3572 | 89.54% | 89.12% |
| 5  | 0.2643 | 0.3390 | 91.80% | 89.66% |
| 6  | 0.2643 | 0.3390 | 91.80% | 89.66% |
| 7  | 0.2202 | 0.3357 | 93.34% | 89.72% |
| 8  | 0.2202 | 0.3357 | 93.34% | 89.72% |
| 9  | 0.1965 | 0.3309 | 94.21% | 89.82% |
| 10 | 0.1965 | 0.3309 | 94.21% | 89.82% |

---

### Experiment 4 — Rank: 4, Alpha: 2, Dropout: 0.1

| Epoch | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy |
|-------|--------------|----------------|------------------|-------------------|
| 1  | 1.1778 | 0.4057 | 73.76% | 87.60% |
| 2  | 1.1778 | 0.4057 | 73.76% | 87.60% |
| 3  | 0.3453 | 0.3595 | 89.45% | 88.64% |
| 4  | 0.3453 | 0.3595 | 89.45% | 88.64% |
| 5  | 0.2770 | 0.3400 | 91.44% | 89.38% |
| 6  | 0.2770 | 0.3400 | 91.44% | 89.38% |
| 7  | 0.2391 | 0.3337 | 92.67% | 89.68% |
| 8  | 0.2391 | 0.3337 | 92.67% | 89.68% |
| 9  | 0.2197 | 0.3299 | 93.40% | 89.74% |
| 10 | 0.2197 | 0.3299 | 93.40% | 89.74% |

---

### Experiment 5 — Rank: 4, Alpha: 4, Dropout: 0.1

| Epoch | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy |
|-------|--------------|----------------|------------------|-------------------|
| 1  | 1.1428 | 0.4044 | 74.14% | 88.00% |
| 2  | 1.1428 | 0.4044 | 74.14% | 88.00% |
| 3  | 0.3398 | 0.3545 | 89.62% | 88.76% |
| 4  | 0.3398 | 0.3545 | 89.62% | 88.76% |
| 5  | 0.2684 | 0.3385 | 91.68% | 89.34% |
| 6  | 0.2684 | 0.3385 | 91.68% | 89.34% |
| 7  | 0.2274 | 0.3339 | 92.97% | 89.64% |
| 8  | 0.2274 | 0.3339 | 92.97% | 89.64% |
| 9  | 0.2061 | 0.3312 | 93.77% | 89.68% |
| 10 | 0.2061 | 0.3312 | 93.77% | 89.68% |

---

### Experiment 6 — Rank: 4, Alpha: 8, Dropout: 0.1

| Epoch | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy |
|-------|--------------|----------------|------------------|-------------------|
| 1  | 1.0935 | 0.4009 | 74.82% | 87.64% |
| 2  | 1.0935 | 0.4009 | 74.82% | 87.64% |
| 3  | 0.3301 | 0.3582 | 89.84% | 88.76% |
| 4  | 0.3301 | 0.3582 | 89.84% | 88.76% |
| 5  | 0.2553 | 0.3397 | 92.02% | 89.38% |
| 6  | 0.2553 | 0.3397 | 92.02% | 89.38% |
| 7  | 0.2106 | 0.3353 | 93.49% | 89.78% |
| 8  | 0.2106 | 0.3353 | 93.49% | 89.78% |
| 9  | 0.1867 | 0.3338 | 94.44% | 89.80% |
| 10 | 0.1867 | 0.3338 | 94.44% | 89.80% |

---

### Experiment 7 — Rank: 8, Alpha: 2, Dropout: 0.1

| Epoch | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy |
|-------|--------------|----------------|------------------|-------------------|
| 1  | 1.2022 | 0.4097 | 72.84% | 87.54% |
| 2  | 1.2022 | 0.4097 | 72.84% | 87.54% |
| 3  | 0.3452 | 0.3610 | 89.46% | 88.82% |
| 4  | 0.3452 | 0.3610 | 89.46% | 88.82% |
| 5  | 0.2768 | 0.3444 | 91.38% | 89.52% |
| 6  | 0.2768 | 0.3444 | 91.38% | 89.52% |
| 7  | 0.2386 | 0.3386 | 92.70% | 89.92% |
| 8  | 0.2386 | 0.3386 | 92.70% | 89.92% |
| 9  | 0.2189 | 0.3349 | 93.42% | 90.00% |
| 10 | 0.2189 | 0.3349 | 93.42% | 90.00% |

---

### Experiment 8 — Rank: 8, Alpha: 4, Dropout: 0.1

| Epoch | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy |
|-------|--------------|----------------|------------------|-------------------|
| 1  | 1.1267 | 0.4078 | 74.37% | 87.62% |
| 2  | 1.1267 | 0.4078 | 74.37% | 87.62% |
| 3  | 0.3382 | 0.3543 | 89.61% | 89.00% |
| 4  | 0.3382 | 0.3543 | 89.61% | 89.00% |
| 5  | 0.2667 | 0.3388 | 91.78% | 89.48% |
| 6  | 0.2667 | 0.3388 | 91.78% | 89.48% |
| 7  | 0.2255 | 0.3322 | 93.12% | 89.56% |
| 8  | 0.2255 | 0.3322 | 93.12% | 89.56% |
| 9  | 0.2034 | 0.3299 | 94.01% | 89.58% |
| 10 | 0.2034 | 0.3299 | 94.01% | 89.58% |

---

### Experiment 9 — Rank: 8, Alpha: 8, Dropout: 0.1

| Epoch | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy |
|-------|--------------|----------------|------------------|-------------------|
| 1  | 1.1006 | 0.4035 | 74.82% | 87.66% |
| 2  | 1.1006 | 0.4035 | 74.82% | 87.66% |
| 3  | 0.3310 | 0.3516 | 89.86% | 88.94% |
| 4  | 0.3310 | 0.3516 | 89.86% | 88.94% |
| 5  | 0.2543 | 0.3434 | 92.13% | 89.68% |
| 6  | 0.2543 | 0.3434 | 92.13% | 89.68% |
| 7  | 0.2095 | 0.3363 | 93.58% | 89.90% |
| 8  | 0.2095 | 0.3363 | 93.58% | 89.90% |
| 9  | 0.1850 | 0.3324 | 94.53% | 90.06% |
| 10 | 0.1850 | 0.3324 | 94.53% | 90.06% |

---

### Optuna Best — Rank: 4, Alpha: 4, Dropout: 0.3, LR: 2.96e-4 ⭐

| Epoch | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy |
|-------|--------------|----------------|------------------|-------------------|
| 1 | 1.0620 | 0.3954 | 75.51% | 88.14% |
| 2 | 0.3245 | 0.3518 | 90.04% | 88.78% |
| 3 | 0.2554 | 0.3351 | 92.01% | 89.46% |
| 4 | 0.2169 | 0.3337 | 93.40% | 89.78% |
| 5 | 0.1949 | 0.3301 | 94.16% | 89.84% |

> **Test Accuracy: 89.73%**

---

### Q1 Final Test Accuracy Table

| LoRA | Rank | Alpha | Dropout | Overall Test Accuracy | Trainable Parameters |
|------|------|-------|---------|----------------------|---------------------|
| No   | —    | —     | —       | 80.46%               | 38,500              |
| Yes  | 2    | 2     | 0.1     | 89.46%               | 93,796              |
| Yes  | 2    | 4     | 0.1     | 89.65%               | 93,796              |
| Yes  | 2    | 8     | 0.1     | 89.58%               | 93,796              |
| Yes  | 4    | 2     | 0.1     | 89.45%               | 149,092             |
| Yes  | 4    | 4     | 0.1     | 89.48%               | 149,092             |
| Yes  | 4    | 8     | 0.1     | **89.85%**           | 149,092             |
| Yes  | 8    | 2     | 0.1     | 89.53%               | 259,684             |
| Yes  | 8    | 4     | 0.1     | 89.57%               | 259,684             |
| Yes  | 8    | 8     | 0.1     | 89.62%               | 259,684             |
| Yes (Optuna) | 4 | 4 | 0.3 | **89.73%**          | 149,092             |

---

### Optuna HPO Summary

Best config: **rank=4, alpha=4, dropout=0.3, lr=0.000296**

| Trial | Rank | Alpha | Dropout | LR | Best Val Acc | Status |
|-------|------|-------|---------|-----|-------------|--------|
| 0  | 16 | 32 | 0.30 | 2.86e-5 | 85.10% | Completed |
| 1  | 4  | 4  | 0.30 | 2.96e-4 | **89.18%** | ✅ Best |
| 2  | 4  | 4  | 0.00 | 1.76e-4 | 89.02% | Completed |
| 3  | 2  | 4  | 0.30 | 8.0e-5  | 84.08% | Pruned |
| 4  | 8  | 4  | 0.10 | 1.17e-4 | 85.68% | Pruned |
| 5  | 2  | 4  | 0.20 | 1.14e-4 | 85.78% | Pruned |
| 6  | 2  | 16 | 0.10 | 8.4e-5  | 84.84% | Pruned |
| 7  | 2  | 16 | 0.10 | 3.11e-4 | 88.94% | Pruned |
| 8  | 2  | 8  | 0.30 | 7.6e-5  | 83.78% | Pruned |
| 9  | 2  | 4  | 0.20 | 1.3e-5  | 16.10% | Pruned |
| 10 | 4  | 2  | 0.20 | 4.61e-4 | 87.94% | Interrupted |

---

## 📊 Q2 Results

### Q2(i) — FGSM Attack: Scratch vs IBM ART

| Metric | Accuracy |
|--------|---------|
| Clean Test Accuracy (ResNet-18) | ≥ 72% |
| Adversarial Accuracy — FGSM **without** IBM ART | **79.02%** |
| Adversarial Accuracy — FGSM **with** IBM ART    | **88.78%** |

> WandB logs with 10 sample images (clean + FGSM scratch + FGSM ART): [Q2 WandB Run](https://wandb.ai/b23bb1043-indian-institute-of-technology-jodhpur/B23BB1043-Assignment5-Q2ii/runs/thpiep6m?nw=nwuserb23bb1043)

---

### Q2(ii) — Adversarial Detection (ResNet-34)

| Attack | Detector Model | Detection Accuracy |
|--------|---------------|-------------------|
| PGD    | ResNet-34     | **88.60%**        |
| BIM    | ResNet-34     | **89.53%**        |

Both detectors exceed the required 70% detection accuracy threshold.

---

## 🐳 Docker Notes

```bash
docker build -t ass5 .
docker run --gpus all -v $(pwd):/workspace -it ass5 bash
cd /workspace
python q1_train.py
python q2i_fgsm.py
python q2ii_detection.py
```
