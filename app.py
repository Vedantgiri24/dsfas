"""
Machine Fault Diagnosis — Streamlit App
Compatible with Python 3.14 (Streamlit Cloud)
Uses PyTorch + torchvision (EfficientNet_B0)
"""

import os
import io
import time
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
DATA_ROOT   = "Split Data"
IMG_SIZE    = 224
BATCH_SIZE  = 16
NUM_CLASSES = 5
SEED        = 42

CLASS_NAMES = [
    "BearingFault",
    "BentShaft",
    "Misalignment",
    "Healthy",
    "FoundationLooseness",
]

CHANNELS = ["CH_1", "CH_2", "CH_4"]

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────
class FaultDataset(Dataset):
    def __init__(self, split: str, channel: str, transform=None):
        self.paths, self.labels = [], []
        self.transform = transform
        split_dir = os.path.join(DATA_ROOT, split)
        for label_idx, cls in enumerate(CLASS_NAMES):
            ch_dir = os.path.join(split_dir, cls, channel)
            if not os.path.isdir(ch_dir):
                continue
            for f in sorted(os.listdir(ch_dir)):
                if f.lower().endswith(".png"):
                    self.paths.append(os.path.join(ch_dir, f))
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────
def build_model(num_classes: int, freeze_base: bool = True):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model.to(DEVICE)


def unfreeze_top_layers(model, n_layers: int = 30):
    params = list(model.parameters())
    for param in params[-n_layers:]:
        param.requires_grad = True
    return model


# ─────────────────────────────────────────────
#  TRAIN / EVAL HELPERS
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out  = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        preds       = out.argmax(1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


def plot_confusion(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Normalised)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
#  STREAMLIT UI
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Machine Fault Diagnosis",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ Machine Fault Diagnosis — CNN Trainer")
st.caption("EfficientNetB0 · Transfer Learning · 5-class vibration signal classification")

# ── Sidebar ──────────────────────────────────
with st.sidebar:
    st.header("Training settings")
    channel     = st.selectbox("Accelerometer channel", CHANNELS)
    epochs_p1   = st.slider("Phase 1 epochs (frozen base)", 5, 30, 15)
    epochs_p2   = st.slider("Phase 2 epochs (fine-tune)", 5, 50, 30)
    lr_p1       = st.select_slider("Learning rate phase 1",
                                   options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
                                   value=1e-4,
                                   format_func=lambda x: f"{x:.0e}")
    lr_p2       = st.select_slider("Learning rate phase 2",
                                   options=[1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
                                   value=1e-5,
                                   format_func=lambda x: f"{x:.0e}")
    batch_size  = st.select_slider("Batch size", [8, 16, 32], value=16)
    st.divider()
    st.caption(f"Device: `{DEVICE}`")

# ── Data check ───────────────────────────────
st.subheader("1 · Dataset overview")

data_ok = os.path.isdir(DATA_ROOT)
if not data_ok:
    st.error(f"Data folder `{DATA_ROOT}` not found. "
             "Make sure `Split Data/` is in the same directory as `app.py`.")
    st.stop()

counts = {}
for split in ["Train", "Val", "Test"]:
    for cls in CLASS_NAMES:
        ch_dir = os.path.join(DATA_ROOT, split, cls, channel)
        n = len([f for f in os.listdir(ch_dir)
                 if f.endswith(".png")]) if os.path.isdir(ch_dir) else 0
        counts[f"{split}/{cls}"] = n

col1, col2, col3 = st.columns(3)
for col, split in zip([col1, col2, col3], ["Train", "Val", "Test"]):
    with col:
        st.markdown(f"**{split}**")
        for cls in CLASS_NAMES:
            st.markdown(f"- {cls}: `{counts[f'{split}/{cls}']}`")

# ── Training ─────────────────────────────────
st.divider()
st.subheader("2 · Train the model")

if "history" not in st.session_state:
    st.session_state.history = None
if "model_state" not in st.session_state:
    st.session_state.model_state = None

run_btn = st.button("Start training", type="primary")

if run_btn:
    train_ds = FaultDataset("Train", channel, train_tf)
    val_ds   = FaultDataset("Val",   channel, val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)

    model     = build_model(NUM_CLASSES, freeze_base=True)
    criterion = nn.CrossEntropyLoss()

    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    best_val_acc = 0.0
    best_state   = None

    progress    = st.progress(0, text="Phase 1 — training head…")
    metrics_box = st.empty()
    chart_ph    = st.empty()

    total_epochs = epochs_p1 + epochs_p2

    # ── Phase 1 ──────────────────────────────
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr_p1
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=4
    )

    for ep in range(epochs_p1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step(vl_loss)

        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        pct = int((ep + 1) / total_epochs * 100)
        progress.progress(pct, text=f"Phase 1 · Epoch {ep+1}/{epochs_p1} "
                          f"— val acc: {vl_acc*100:.1f}%")
        metrics_box.markdown(
            f"**Epoch {ep+1}** &nbsp;|&nbsp; "
            f"Train acc `{tr_acc*100:.1f}%` &nbsp;|&nbsp; "
            f"Val acc `{vl_acc*100:.1f}%` &nbsp;|&nbsp; "
            f"Val loss `{vl_loss:.4f}`"
        )

    # ── Phase 2 ──────────────────────────────
    model = unfreeze_top_layers(model, 30)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr_p2
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5
    )

    patience_count = 0
    for ep in range(epochs_p2):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step(vl_loss)

        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)

        if vl_acc > best_val_acc:
            best_val_acc   = vl_acc
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        pct = int((epochs_p1 + ep + 1) / total_epochs * 100)
        progress.progress(pct, text=f"Phase 2 · Epoch {ep+1}/{epochs_p2} "
                          f"— best val acc: {best_val_acc*100:.1f}%")
        metrics_box.markdown(
            f"**Phase 2 · Epoch {ep+1}** &nbsp;|&nbsp; "
            f"Train acc `{tr_acc*100:.1f}%` &nbsp;|&nbsp; "
            f"Val acc `{vl_acc*100:.1f}%` &nbsp;|&nbsp; "
            f"Best `{best_val_acc*100:.1f}%`"
        )

        if patience_count >= 12:
            st.info("Early stopping triggered.")
            break

    progress.progress(100, text="Training complete!")
    st.session_state.history      = history
    st.session_state.model_state  = best_state
    st.session_state.best_val_acc = best_val_acc
    st.success(f"Best val accuracy: **{best_val_acc*100:.2f}%**")

# ── Results ───────────────────────────────────
if st.session_state.history:
    st.divider()
    st.subheader("3 · Training curves")

    h = st.session_state.history
    ep_range = list(range(1, len(h["train_acc"]) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(ep_range, [a*100 for a in h["train_acc"]], label="Train")
    ax1.plot(ep_range, [a*100 for a in h["val_acc"]],   label="Val")
    ax1.axvline(x=epochs_p1, color="gray", linestyle="--",
                linewidth=0.8, label="Phase 2 start")
    ax1.set_title("Accuracy (%)")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(ep_range, h["train_loss"], label="Train")
    ax2.plot(ep_range, h["val_loss"],   label="Val")
    ax2.axvline(x=epochs_p1, color="gray", linestyle="--",
                linewidth=0.8, label="Phase 2 start")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Test evaluation ───────────────────────────
if st.session_state.model_state:
    st.divider()
    st.subheader("4 · Test set evaluation")

    if st.button("Evaluate on test set"):
        test_ds     = FaultDataset("Test", channel, val_tf)
        test_loader = DataLoader(test_ds, batch_size=batch_size,
                                 shuffle=False, num_workers=0)

        eval_model = build_model(NUM_CLASSES, freeze_base=False)
        eval_model.load_state_dict(st.session_state.model_state)
        eval_model.to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        t_loss, t_acc, preds, labels = evaluate(eval_model, test_loader, criterion)

        col1, col2 = st.columns(2)
        col1.metric("Test accuracy", f"{t_acc*100:.2f}%")
        col2.metric("Test loss",     f"{t_loss:.4f}")

        st.text(classification_report(labels, preds, target_names=CLASS_NAMES))

        fig = plot_confusion(labels, preds, CLASS_NAMES)
        st.pyplot(fig)
        plt.close()

        # Download button for model weights
        buf = io.BytesIO()
        torch.save(st.session_state.model_state, buf)
        buf.seek(0)
        st.download_button(
            label="Download model weights (.pt)",
            data=buf,
            file_name="best_fault_model.pt",
            mime="application/octet-stream",
        )

# ── Single image inference ────────────────────
if st.session_state.model_state:
    st.divider()
    st.subheader("5 · Single image inference")

    uploaded = st.file_uploader("Upload a signal image (PNG)", type=["png"])
    if uploaded:
        img_pil = Image.open(uploaded).convert("RGB")
        st.image(img_pil, caption="Uploaded signal", use_container_width=True)

        tensor = val_tf(img_pil).unsqueeze(0).to(DEVICE)

        inf_model = build_model(NUM_CLASSES, freeze_base=False)
        inf_model.load_state_dict(st.session_state.model_state)
        inf_model.to(DEVICE).eval()

        with torch.no_grad():
            logits = inf_model(tensor)
            probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        pred_idx  = probs.argmax()
        pred_name = CLASS_NAMES[pred_idx]
        confidence = probs[pred_idx] * 100

        st.success(f"Predicted fault: **{pred_name}** ({confidence:.1f}% confidence)")

        fig, ax = plt.subplots(figsize=(7, 3))
        bars = ax.barh(CLASS_NAMES, probs * 100, color="steelblue")
        bars[pred_idx].set_color("darkorange")
        ax.set_xlabel("Probability (%)")
        ax.set_xlim(0, 100)
        ax.set_title("Class probabilities")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()