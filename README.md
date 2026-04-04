# Assignment 5 – ViT Fine-tuning with LoRA & Adversarial Attacks

**Name:** Tushar Yadav  
**Roll No:** B23BB1043  

---

## 🔗 Links

| Resource | Link |
|----------|------|
| WandB Project (Q1) | _[Add after running]_ |
| WandB Project (Q2) | _[Add after running]_ |
| HuggingFace Model  | _[Add after running]_ |

---

## 📁 Repository Structure

```
Assignment-5/
├── q1_train.py          # Q1: ViT-S fine-tuning (no-LoRA + all LoRA combos)
├── q1_optuna.py         # Q1: Optuna HPO for LoRA hyperparameters
├── q1_upload_hf.py      # Q1: Push best model to HuggingFace Hub
├── q2i_fgsm.py          # Q2(i): ResNet-18 training + FGSM (scratch vs ART)
├── q2ii_detection.py    # Q2(ii): Adversarial Detection (PGD & BIM)
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚙️ Installation

### Option 1 – Docker (recommended, as required by assignment)

```bash
# Build image
docker build -t ass5 .

# Run with GPU
docker run --gpus all -it ass5 bash
```

### Option 2 – Google Colab / Local pip

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Q1 – ViT-S Fine-tuning on CIFAR-100

#### Step 1: Run all experiments (no-LoRA + 9 LoRA combos)
```bash
python q1_train.py
```
This will:
- Train no-LoRA baseline (classification head only)
- Train all 9 LoRA combinations (rank ∈ {2,4,8}, alpha ∈ {2,4,8}, dropout=0.1)
- Log everything to WandB
- Save best checkpoints to `checkpoints/`

#### Step 2: Optuna hyperparameter search
```bash
python q1_optuna.py
```
This searches over rank, alpha, dropout, and LR using Optuna (20 trials) and re-trains the best config.

#### Step 3: Upload best model to HuggingFace
```bash
# For LoRA model
python q1_upload_hf.py \
  --model_path checkpoints/LoRA_r<R>_a<A>_d1_best.pt \
  --run_name LoRA_r<R>_a<A>_d1 \
  --hf_repo TusharYadav/vit-cifar100-lora

# For no-LoRA model
python q1_upload_hf.py \
  --model_path checkpoints/NoLoRA_best.pt \
  --run_name NoLoRA \
  --hf_repo TusharYadav/vit-cifar100-no-lora \
  --no_lora
```

---

### Q2 – Adversarial Attacks

#### Q2(i): FGSM Attack (scratch vs ART)
```bash
python q2i_fgsm.py
```
This will:
- Train ResNet-18 on CIFAR-10 from scratch (target ≥ 72% test accuracy)
- Run FGSM from scratch
- Run FGSM via IBM ART
- Compare accuracy at multiple epsilon values
- Log 10 sample images to WandB

#### Q2(ii): Adversarial Detection
```bash
# Must run Q2(i) first (needs checkpoints/resnet18_cifar10_best.pt)
python q2ii_detection.py
```
This will:
- Generate PGD and BIM adversarial examples using ART
- Train ResNet-34 binary detector on clean + PGD
- Train ResNet-34 binary detector on clean + BIM
- Log detection accuracy and comparison chart to WandB

---

## 📊 Q1 Results

### Training Table (example – fill in after running)

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|----------|-----------|---------|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |

### Testing Table

| LoRA | Rank | Alpha | Dropout | Test Acc | Trainable Params |
|------|------|-------|---------|----------|-----------------|
| No   | –    | –     | –       |          |                 |
| Yes  | 2    | 2     | 0.1     |          |                 |
| Yes  | 2    | 4     | 0.1     |          |                 |
| Yes  | 2    | 8     | 0.1     |          |                 |
| Yes  | 4    | 2     | 0.1     |          |                 |
| Yes  | 4    | 4     | 0.1     |          |                 |
| Yes  | 4    | 8     | 0.1     |          |                 |
| Yes  | 8    | 2     | 0.1     |          |                 |
| Yes  | 8    | 4     | 0.1     |          |                 |
| Yes  | 8    | 8     | 0.1     |          |                 |
| Yes  | Best (Optuna) | | 0.1 | | |

---

## 📊 Q2 Results

### Q2(i) FGSM Comparison

| ε     | Clean Acc | FGSM Scratch | FGSM ART |
|-------|-----------|--------------|----------|
| 0.01  |           |              |          |
| 0.02  |           |              |          |
| 0.03  |           |              |          |
| 0.05  |           |              |          |
| 0.10  |           |              |          |

### Q2(ii) Detection Accuracy

| Attack | Detector | Detection Acc |
|--------|----------|--------------|
| PGD    | ResNet-34 |              |
| BIM    | ResNet-34 |              |

---

## 🐳 Docker Notes

The assignment requires running inside Docker. The `Dockerfile` uses `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime` as the base image. GPU is required for practical training times.

To run in Colab with Docker-like environment:
```python
!pip install -r requirements.txt
!python q1_train.py
```
