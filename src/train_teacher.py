# ============================================================
# PolyTAO Teacher Training Script
# CLEAN — Lightning-consistent logging + working plots
# ============================================================

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput

# ============================================================
# Dataset
# ============================================================

class BigSMILESDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=128):
        df = pd.read_csv(csv_path)

        self.texts = df["product"].astype(str).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.property_cols = [
            "MolWt", "HeavyAtomCount", "NHOHCount", "NOCount",
            "NumAliphaticCarbocycles", "NumAliphaticHeterocycles",
            "NumAliphaticRings", "NumAromaticCarbocycles",
            "NumAromaticHeterocycles", "NumAromaticRings",
            "NumHAcceptors", "NumHDonors", "NumHeteroatoms",
            "NumRotatableBonds", "RingCount",
        ]

        props = []
        for col in self.property_cols:
            norm_col = f"{col}_norm"
            props.append(df[norm_col].values if norm_col in df.columns else df[col].values)

        self.properties = np.stack(props, axis=1).astype("float32")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        labels = enc["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "properties": torch.tensor(self.properties[idx], dtype=torch.float32)
        }

# ============================================================
# Teacher LightningModule
# ============================================================

class PolyTAOTeacherPL(pl.LightningModule):
    def __init__(self, model_name, lr=1e-5):
        super().__init__()
        self.save_hyperparameters()

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.property_proj = nn.Linear(15, self.model.config.d_model)

    def forward(self, input_ids, attention_mask, labels=None, properties=None):
        enc_out = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        props = properties.clamp(-10.0, 10.0)
        prop_emb = self.property_proj(props)
        prop_emb = prop_emb / prop_emb.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        prop_emb = 0.05 * prop_emb.unsqueeze(1)

        hidden = enc_out.last_hidden_state + prop_emb
        enc_out = BaseModelOutput(last_hidden_state=hidden)

        return self.model(
            encoder_outputs=enc_out,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

    def training_step(self, batch, batch_idx):
        out = self(**batch)
        self.log("train/loss", out.loss, on_step=True, on_epoch=True)
        return out.loss

    def validation_step(self, batch, batch_idx):
        out = self(**batch)
        self.log("val/loss", out.loss, on_step=False, on_epoch=True)
        return out.loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

# ============================================================
# Main
# ============================================================

def main():
    BASE_DIR = "/content/drive/MyDrive/polytao_thesis"

    DATA_PATH = os.path.join(BASE_DIR, "data", "polymers_with_properties_normalized_train.csv")

    TEACHER_DIR = os.path.join(BASE_DIR, "checkpoints", "teacher")
    MODEL_DIR = os.path.join(TEACHER_DIR, "model")
    LOG_DIR = os.path.join(TEACHER_DIR, "logs")
    PLOT_DIR = os.path.join(TEACHER_DIR, "plots")

    for d in [MODEL_DIR, LOG_DIR, PLOT_DIR]:
        os.makedirs(d, exist_ok=True)

    MODEL_NAME = "hkqiu/PolyTAO-BigSMILES_Version"

    BATCH_SIZE = 8
    MAX_LEN = 128
    EPOCHS = 3
    LR = 1e-5
    VAL_RATIO = 0.1

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = BigSMILESDataset(DATA_PATH, tokenizer, MAX_LEN)

    train_size = int((1 - VAL_RATIO) * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    teacher = PolyTAOTeacherPL(MODEL_NAME, LR)

    logger = CSVLogger(save_dir=LOG_DIR, name="")

    trainer = Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="32",
        logger=logger,
        log_every_n_steps=1,
        enable_checkpointing=False,
        enable_model_summary=False
    )

    trainer.fit(teacher, train_loader, val_loader)

    teacher.model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    torch.save(
        teacher.property_proj.state_dict(),
        os.path.join(MODEL_DIR, "property_proj.pt")
    )

    # ====================================================
    # Plot loss curves (CORRECT COLUMN NAMES)
    # ====================================================

    metrics_path = os.path.join(logger.log_dir, "metrics.csv")
    df = pd.read_csv(metrics_path)

    # ---- train loss (epoch) ----
    train_df = df[df["train/loss_epoch"].notna()]
    train_df = train_df.groupby("epoch", as_index=False)["train/loss_epoch"].mean()

    plt.figure(figsize=(6,4))
    plt.plot(train_df["epoch"], train_df["train/loss_epoch"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Teacher Training Loss")
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, "teacher_train_loss.png"), dpi=200)
    plt.close()

    # ---- validation loss ----
    val_df = df[df["val/loss"].notna()]
    val_df = val_df.groupby("epoch", as_index=False)["val/loss"].mean()

    plt.figure(figsize=(6,4))
    plt.plot(val_df["epoch"], val_df["val/loss"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Teacher Validation Loss")
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, "teacher_val_loss.png"), dpi=200)
    plt.close()

if __name__ == "__main__":
    main()
