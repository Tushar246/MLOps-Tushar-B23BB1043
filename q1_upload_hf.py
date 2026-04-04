"""
Q1 – Push best LoRA model weights to HuggingFace Hub
Author: Tushar Yadav | Roll No: B23BB1043

Usage:
    python q1_upload_hf.py --model_path checkpoints/<run_name>_best.pt \
                           --run_name <run_name> \
                           --hf_repo TusharYadav/vit-cifar100-lora
"""

import argparse
import os
import torch
from transformers import ViTForImageClassification
from peft import PeftModel, LoraConfig, get_peft_model
from huggingface_hub import HfApi, login

MODEL_NAME  = "WinKawaks/vit-small-patch16-224"
NUM_CLASSES = 100


def push_lora_model(model_path: str, run_name: str, hf_repo: str):
    login()                      # prompts for HF token if not cached
    api = HfApi()

    print(f"Loading base ViT-S …")
    base = ViTForImageClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True
    )

    print(f"Loading PEFT adapter from {model_path} …")
    model = PeftModel.from_pretrained(base, model_path)

    repo_id = hf_repo
    try:
        api.create_repo(repo_id, private=False, exist_ok=True)
    except Exception as e:
        print(f"Repo creation note: {e}")

    print(f"Pushing to HuggingFace: {repo_id} …")
    model.push_to_hub(repo_id, commit_message=f"Upload {run_name} PEFT LoRA weights")
    print("Done ✓")


def push_no_lora_model(model_path: str, hf_repo: str):
    login()
    api = HfApi()

    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True
    )
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)

    try:
        api.create_repo(hf_repo, private=False, exist_ok=True)
    except Exception:
        pass

    model.push_to_hub(hf_repo, commit_message="Upload NoLoRA classification head weights")
    print("Done ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--run_name",   required=True)
    parser.add_argument("--hf_repo",    default="TusharYadav/vit-cifar100-lora")
    parser.add_argument("--no_lora",    action="store_true")
    args = parser.parse_args()

    if args.no_lora:
        push_no_lora_model(args.model_path, args.hf_repo)
    else:
        push_lora_model(args.model_path, args.run_name, args.hf_repo)
