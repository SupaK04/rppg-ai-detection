import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.models.bi_arterial_net import BiArterialDeepfakeNet
from src.models.losses import pearson_correlation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TensorDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        tf = torch.from_numpy(item["forehead"]).float()
        tc = torch.from_numpy(item["cheek"]).float()
        lbl = torch.tensor(item["label"]).long()
        return tf, tc, lbl


def main():
    print("=" * 80)
    print("  Dual-Stream Bi-Arterial Deepfake Network: Training on Modern AI Generative Suite")
    print("  Compute Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    print("=" * 80)

    data_file = os.path.join(BASE_DIR, "scratch", "training_cache", "cached_large_dataset.pt")
    assert os.path.exists(data_file), f"Dataset file not found: {data_file}. Run cache_large_dataset.py first."

    all_data = torch.load(data_file, weights_only=False)
    n_real = sum(1 for d in all_data if d["label"] == 1)
    n_fake = sum(1 for d in all_data if d["label"] == 0)
    print(f"Loaded {len(all_data)} total segments from database ({n_real} Real, {n_fake} Fake).")

    # Stratified Train/Val split (80/20)
    labels = [d["label"] for d in all_data]
    train_data, val_data = train_test_split(all_data, test_size=0.20, stratify=labels, random_state=42)
    print(f"Split: {len(train_data)} Training samples, {len(val_data)} Validation samples.")

    train_loader = DataLoader(TensorDataset(train_data), batch_size=4, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(val_data), batch_size=4, shuffle=False)

    model = BiArterialDeepfakeNet().to(device)

    # Initialize 3D-CNN feature branch with pretrained Siamese-rPPG weights
    pretrained_path = os.path.join(BASE_DIR, "models", "pretrained_siamese_rppg.pth")
    if os.path.exists(pretrained_path):
        state_dict = torch.load(pretrained_path, map_location=device, weights_only=True)
        branch_dict = {}
        for k, v in state_dict.items():
            if k.startswith("shared_branch."):
                branch_dict[k.replace("shared_branch.", "")] = v
        if branch_dict:
            model.branch.load_state_dict(branch_dict)
            print("✓ Initialized 3D-CNN branch with pretrained UBFC physiological weights.")

    criterion_ce = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    num_epochs = 25
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    best_val_f1 = 0.0
    best_epoch = 0
    save_path = os.path.join(BASE_DIR, "models", "bi_arterial_deepfake_net.pth")

    start_time = time.time()

    print(f"\nStarting {num_epochs} Epochs Training Loop...")
    print("-" * 80)

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for tf, tc, lbl in train_loader:
            tf, tc, lbl = tf.to(device), tc.to(device), lbl.to(device)
            optimizer.zero_grad()

            logits, sig_f, sig_c = model(tf, tc)

            # 1. Classification CrossEntropy Loss
            loss_ce = criterion_ce(logits, lbl)

            # 2. Biological Synchrony Regularization
            r = pearson_correlation(sig_f, sig_c)  # (B,)
            lbl_float = lbl.float()
            loss_real_sync = torch.mean((1.0 - r) * lbl_float)
            loss_fake_supp = torch.mean(torch.relu(r + 0.05) ** 2 * (1.0 - lbl_float))
            loss_bio = loss_real_sync + 0.8 * loss_fake_supp

            loss = loss_ce + 0.40 * loss_bio

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = train_loss / max(1, n_batches)

        # Validation Phase
        model.eval()
        val_preds, val_targets, val_probs = [], [], []

        with torch.no_grad():
            for tf, tc, lbl in val_loader:
                tf, tc, lbl = tf.to(device), tc.to(device), lbl.to(device)
                logits, _, _ = model(tf, tc)
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = torch.argmax(logits, dim=1)

                val_preds.extend(preds.cpu().numpy().tolist())
                val_targets.extend(lbl.cpu().numpy().tolist())
                val_probs.extend(probs.cpu().numpy().tolist())

        val_acc = accuracy_score(val_targets, val_preds) * 100
        val_f1 = f1_score(val_targets, val_preds, zero_division=0) * 100
        try:
            val_auc = roc_auc_score(val_targets, val_probs) * 100
        except Exception:
            val_auc = 0.0

        print(f"Epoch [{epoch:2d}/{num_epochs:2d}] | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:5.1f}% | Val F1: {val_f1:5.1f}% | AUC: {val_auc:5.1f}%")

        if val_f1 >= best_val_f1 and val_acc >= 75.0:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_auc": val_auc
            }, save_path)

    elapsed = time.time() - start_time
    print("-" * 80)
    print(f"✓ Training finished in {elapsed:.1f}s.")
    print(f"✓ Best Model (Epoch {best_epoch}): Val F1 = {best_val_f1:.1f}% saved to: {save_path}")


if __name__ == "__main__":
    main()
