"""
BERT Siamese Baseline for Authorship Verification
===================================================
Fine-tunes bert-base-uncased as a Siamese network for authorship
verification. Demonstrates the accuracy-robustness trade-off holds
even for transformer-based models.

Architecture:
  - Shared BERT encoder → [CLS] pooling
  - Interaction layer: |u-v|, u*v concatenation
  - MLP classifier head

Usage:
    # Quick validation (CPU/MPS, ~5 min)
    python experiments/train_bert_baseline.py --max_pairs 200 --epochs 2

    # Full training (GPU recommended, ~2-3 hrs)
    python experiments/train_bert_baseline.py --max_pairs 5000 --epochs 5
"""
import sys
import os
import json
import re
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from transformers import BertTokenizer, BertModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_loader_scie import PAN22Loader, BlogTextLoader, EnronLoader

# ==============================================================================
# Configuration
# ==============================================================================
DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
SEED = 42
OUTPUT_DIR = "results/bert_baseline"

def parse_args():
    parser = argparse.ArgumentParser(description="BERT Siamese Baseline")
    parser.add_argument("--max_pairs", type=int, default=2000,
                        help="Max training pairs from PAN22 (default 2000)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Training epochs (default 5)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (default 16)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate (default 2e-5)")
    parser.add_argument("--max_len", type=int, default=256,
                        help="Max token length per text (default 256)")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training, only evaluate saved model")
    return parser.parse_args()


# ==============================================================================
# Dataset
# ==============================================================================
class BERTPairDataset(Dataset):
    """Tokenizes text pairs on-the-fly for BERT Siamese."""

    def __init__(self, texts1, texts2, labels, tokenizer, max_len=256):
        self.texts1 = texts1
        self.texts2 = texts2
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc1 = self.tokenizer(
            self.texts1[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        enc2 = self.tokenizer(
            self.texts2[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids_1': enc1['input_ids'].squeeze(0),
            'attention_mask_1': enc1['attention_mask'].squeeze(0),
            'input_ids_2': enc2['input_ids'].squeeze(0),
            'attention_mask_2': enc2['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.float32),
        }


# ==============================================================================
# Model
# ==============================================================================
class BERTSiamese(nn.Module):
    """
    Siamese BERT for Authorship Verification.

    Architecture:
      - Shared BERT (bert-base-uncased) encoder
      - [CLS] token as sentence embedding (768-dim)
      - Interaction: [u, v, |u-v|, u*v] → 768*4 = 3072-dim
      - MLP head: 3072 → 512 → 128 → 1
    """

    def __init__(self, bert_model_name='bert-base-uncased', freeze_bert_layers=8):
        super().__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)

        # Freeze lower layers for efficiency (keep top layers trainable)
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < freeze_bert_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        # Also freeze embeddings
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        hidden_size = self.bert.config.hidden_size  # 768

        self.head = nn.Sequential(
            nn.Linear(hidden_size * 4, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def encode(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return cls_output

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        u = self.encode(input_ids_1, attention_mask_1)
        v = self.encode(input_ids_2, attention_mask_2)

        diff = torch.abs(u - v)
        prod = u * v
        combined = torch.cat([u, v, diff, prod], dim=1)

        logits = self.head(combined)
        return logits


# ==============================================================================
# Data Preparation
# ==============================================================================
def preprocess_text(text):
    """Clean text for BERT input."""
    text = str(text)
    text = text.replace("<nl>", " ")
    text = text.replace("<new>", " ")
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:1000]  # Limit to ~1000 chars (BERT handles tokenization)


def load_pan22_pairs(max_pairs=2000):
    """Load PAN22 pairs for training."""
    print(f"Loading PAN22 data (max {max_pairs} pairs)...")
    loader = PAN22Loader(
        "pan22-authorship-verification-training.jsonl",
        "pan22-authorship-verification-training-truth.jsonl"
    )
    loader.load(limit=max_pairs * 3)  # Load more to get enough pairs
    t1, t2, labels = loader.create_pairs(num_pairs=max_pairs)

    labels = np.array(labels, dtype=np.float32)
    valid = labels != -1
    t1 = [preprocess_text(t1[i]) for i in range(len(t1)) if valid[i]]
    t2 = [preprocess_text(t2[i]) for i in range(len(t2)) if valid[i]]
    labels = labels[valid]

    print(f"  Loaded {len(labels)} valid pairs "
          f"(pos={int(labels.sum())}, neg={int(len(labels) - labels.sum())})")
    return t1, t2, labels


def load_eval_domain(domain_name, max_pairs=500):
    """Load evaluation pairs for a domain."""
    loaders = {
        'PAN22': lambda: PAN22Loader(
            "pan22-authorship-verification-training.jsonl",
            "pan22-authorship-verification-training-truth.jsonl"),
        'Blog': lambda: BlogTextLoader("blogtext.csv"),
        'Enron': lambda: EnronLoader("emails.csv"),
    }

    if domain_name not in loaders:
        return [], [], np.array([])

    loader = loaders[domain_name]()
    loader.load(limit=6000)
    t1, t2, labels = loader.create_pairs(num_pairs=max_pairs)

    labels = np.array(labels, dtype=np.float32)
    valid = labels != -1
    t1 = [preprocess_text(t1[i]) for i in range(len(t1)) if valid[i]]
    t2 = [preprocess_text(t2[i]) for i in range(len(t2)) if valid[i]]
    labels = labels[valid]

    return t1, t2, labels


# ==============================================================================
# Training
# ==============================================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="  Training", leave=False):
        input_ids_1 = batch['input_ids_1'].to(device)
        attention_mask_1 = batch['attention_mask_1'].to(device)
        input_ids_2 = batch['input_ids_2'].to(device)
        attention_mask_2 = batch['attention_mask_2'].to(device)
        labels = batch['label'].to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy().reshape(-1))

    avg_loss = total_loss / len(loader)
    preds_bin = [1 if p > 0.5 else 0 for p in all_preds]
    acc = accuracy_score(all_labels, preds_bin)
    return avg_loss, acc


def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Evaluating", leave=False):
            input_ids_1 = batch['input_ids_1'].to(device)
            attention_mask_1 = batch['attention_mask_1'].to(device)
            input_ids_2 = batch['input_ids_2'].to(device)
            attention_mask_2 = batch['attention_mask_2'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)

            logits = model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy().reshape(-1))

    avg_loss = total_loss / max(len(loader), 1)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds_bin = (all_probs > 0.5).astype(int)

    acc = accuracy_score(all_labels, preds_bin)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0
    f1 = f1_score(all_labels, preds_bin, zero_division=0)

    return avg_loss, acc, auc, f1, all_probs, all_labels


# ==============================================================================
# Cross-Domain Evaluation
# ==============================================================================
def evaluate_cross_domain(model, tokenizer, args, device):
    """Evaluate BERT on all domains."""
    print("\n" + "=" * 60)
    print("CROSS-DOMAIN EVALUATION")
    print("=" * 60)

    results = {}
    criterion = nn.BCEWithLogitsLoss()

    for domain in ['PAN22', 'Blog', 'Enron']:
        print(f"\n  {domain}...")
        t1, t2, labels = load_eval_domain(domain, max_pairs=500)
        if len(labels) == 0:
            continue

        dataset = BERTPairDataset(t1, t2, labels, tokenizer, args.max_len)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0)

        _, acc, auc, f1, _, _ = evaluate_model(model, loader, criterion, device)

        results[domain] = {
            'accuracy': round(acc, 4),
            'auc': round(auc, 4),
            'f1': round(f1, 4),
            'n_pairs': len(labels),
        }
        print(f"    Acc={acc:.4f}  AUC={auc:.4f}  F1={f1:.4f}  (n={len(labels)})")

    # Average
    if results:
        avg_acc = np.mean([r['accuracy'] for r in results.values()])
        avg_auc = np.mean([r['auc'] for r in results.values()])
        avg_f1 = np.mean([r['f1'] for r in results.values()])
        results['Average'] = {
            'accuracy': round(avg_acc, 4),
            'auc': round(avg_auc, 4),
            'f1': round(avg_f1, 4),
        }
        print(f"\n  Average: Acc={avg_acc:.4f}  AUC={avg_auc:.4f}  F1={avg_f1:.4f}")

    return results


# ==============================================================================
# Main
# ==============================================================================
def main():
    args = parse_args()
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("BERT SIAMESE BASELINE")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Max pairs: {args.max_pairs}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Max sequence length: {args.max_len}")

    # Load tokenizer
    print("\nLoading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    if not args.eval_only:
        # Load training data
        t1, t2, labels = load_pan22_pairs(args.max_pairs)

        # Split
        indices = list(range(len(labels)))
        idx_train, idx_val = train_test_split(
            indices, test_size=0.2, random_state=SEED, stratify=labels)

        t1_train = [t1[i] for i in idx_train]
        t2_train = [t2[i] for i in idx_train]
        y_train = labels[idx_train]

        t1_val = [t1[i] for i in idx_val]
        t2_val = [t2[i] for i in idx_val]
        y_val = labels[idx_val]

        print(f"Train: {len(y_train)} pairs | Val: {len(y_val)} pairs")

        # Create datasets
        train_ds = BERTPairDataset(t1_train, t2_train, y_train, tokenizer, args.max_len)
        val_ds = BERTPairDataset(t1_val, t2_val, y_val, tokenizer, args.max_len)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=0)

        print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

        # Init model
        print("\nInitializing BERT Siamese...")
        model = BERTSiamese(freeze_bert_layers=8).to(DEVICE)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,} "
              f"({100*trainable_params/total_params:.1f}%)")

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=1e-2
        )

        # Training loop
        best_val_acc = 0
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

        print("\nStarting Training...")
        for epoch in range(args.epochs):
            t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            v_loss, v_acc, v_auc, v_f1, _, _ = evaluate_model(
                model, val_loader, criterion, DEVICE)

            history['train_loss'].append(t_loss)
            history['val_loss'].append(v_loss)
            history['train_acc'].append(t_acc)
            history['val_acc'].append(v_acc)

            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | "
                  f"Val Loss: {v_loss:.4f} Acc: {v_acc:.4f} AUC: {v_auc:.4f} F1: {v_f1:.4f}")

            if v_acc > best_val_acc:
                best_val_acc = v_acc
                torch.save(model.state_dict(), f"{OUTPUT_DIR}/bert_siamese_best.pth")
                print(f"  → Saved best model (Acc={v_acc:.4f})")

        # Load best model for evaluation
        model.load_state_dict(torch.load(f"{OUTPUT_DIR}/bert_siamese_best.pth",
                                          map_location=DEVICE))
        print(f"\nBest validation accuracy: {best_val_acc:.4f}")

        # Save training history
        with open(f"{OUTPUT_DIR}/training_history.json", 'w') as f:
            json.dump(history, f, indent=2)

    else:
        # Eval-only mode
        model_path = f"{OUTPUT_DIR}/bert_siamese_best.pth"
        if not os.path.exists(model_path):
            print(f"No saved model found at {model_path}")
            return
        model = BERTSiamese(freeze_bert_layers=8).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Loaded model from {model_path}")

    # Cross-domain evaluation
    eval_results = evaluate_cross_domain(model, tokenizer, args, DEVICE)

    # Save results
    output = {
        'model': 'BERT Siamese (bert-base-uncased)',
        'architecture': {
            'encoder': 'bert-base-uncased (768-dim [CLS])',
            'frozen_layers': 8,
            'interaction': '[u, v, |u-v|, u*v] → 3072-dim',
            'head': '3072 → 512 → 128 → 1',
        },
        'training': {
            'dataset': 'PAN22',
            'max_pairs': args.max_pairs,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'max_len': args.max_len,
        },
        'results': eval_results,
    }

    results_path = f"{OUTPUT_DIR}/bert_baseline_results.json"
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {results_path}")
    print(f"Model saved to {OUTPUT_DIR}/bert_siamese_best.pth")
    print(f"{'='*60}")

    return output


if __name__ == "__main__":
    main()
