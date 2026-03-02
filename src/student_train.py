# ============================================================
# student_train_lightning.py
# FINAL — Property-Conditioned KD + Loss Logging (IU-ready)
# ============================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from torch.utils.data import DataLoader, random_split  
from train_teacher import BigSMILESDataset

# ----------------------------
# CONFIG
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = "/content/drive/MyDrive/polytao_thesis"
DATA_PATH = f"{BASE_DIR}/data/polymers_with_properties_normalized_train.csv"

TEACHER_PATH = f"{BASE_DIR}/checkpoints/teacher_final"
OUT_BASE = f"{BASE_DIR}/checkpoints/polytao_student_distilled_improved"

STUDENT_SCALES = [50]
VAL_RATIO = 0.1

EPOCHS = 3
BATCH_SIZE = 16
MAX_LEN = 160

LR = 5e-5 #learning rate
ALPHA = 0.5
PROPERTY_SCALE = 0.05

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
# Teacher wrapper (conditioned encoder)
# ============================================================
class ConditionedTeacher(nn.Module):
    def __init__(self, teacher_model: T5ForConditionalGeneration, teacher_prop_proj: nn.Linear):
        super().__init__()
        self.teacher = teacher_model
        self.prop_proj = teacher_prop_proj

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        for p in self.prop_proj.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, input_ids, attention_mask, properties):
        enc_out = self.teacher.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        props = properties.clamp(-10.0, 10.0)
        prop_emb = self.prop_proj(props)
        prop_emb = prop_emb / prop_emb.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        prop_emb = PROPERTY_SCALE * prop_emb.unsqueeze(1)

        return enc_out.last_hidden_state + prop_emb


# ============================================================
# Student LightningModule
# ============================================================
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

# ===============================
# Hyperparams (global or config)
# ===============================

class PolyTAOStudentPL(pl.LightningModule):

    def __init__(
        self,
        teacher_ckpt_path: str,
        capacity_percent: int,
        lr: float = 3e-5
    ):
        super().__init__()
        self.save_hyperparameters()

        # --------------------------------------------------
        # Load teacher (frozen)
        # --------------------------------------------------
        teacher = T5ForConditionalGeneration.from_pretrained(
            teacher_ckpt_path,
            output_hidden_states=True
        )
        teacher_cfg = teacher.config

        # --------------------------------------------------
        # Build student config (THIS IS WHERE % IS APPLIED)
        # --------------------------------------------------
        def scale_layers(n, cap):
            return max(1, int(round(n * cap / 100)))
        

        student_cfg = copy.deepcopy(teacher_cfg)

        # Scale encoder layers 
        student_cfg.num_layers = scale_layers(
             teacher_cfg.num_layers, capacity_percent
            )
        
        # Scale decoder layers 
        student_cfg.num_decoder_layers = scale_layers(
            teacher_cfg.num_decoder_layers, capacity_percent
           )

        # 🔹 Important: student must NOT reuse teacher weights
        self.student = T5ForConditionalGeneration(student_cfg)

        # --------------------------------------------------
        # Property projection (teacher → student)
        # --------------------------------------------------
        teacher_prop_proj = nn.Linear(15, teacher_cfg.d_model)
        teacher_prop_proj.load_state_dict(
            torch.load(
                os.path.join(teacher_ckpt_path, "property_proj.pt"),
                map_location="cpu"
            )
        )

        self.property_proj = nn.Linear(15, self.student.config.d_model)
        self.property_proj.load_state_dict(teacher_prop_proj.state_dict())

        # --------------------------------------------------
        # Teacher wrapper (frozen)
        # --------------------------------------------------
        self.teacher = ConditionedTeacher(
            teacher,
            teacher_prop_proj
        )
        for p in self.teacher.parameters():
            p.requires_grad = False

        # --------------------------------------------------
        # KD projection (teacher → student dim)
        # --------------------------------------------------
        self.teacher_to_student_proj = nn.Linear(
            teacher_cfg.d_model,
            self.student.config.d_model
        )

        # --------------------------------------------------
        # Sanity print (VERY IMPORTANT)
        # --------------------------------------------------
        print(
            f"✅ Student initialized at {capacity_percent}% capacity | "
            f"Encoder layers: {self.student.config.num_layers} | "
            f"Decoder layers: {self.student.config.num_decoder_layers}"
        )



    # ==================================================
    # Conditioned encoder forward
    # ==================================================
    def forward_student_conditioned_encoder(self, batch):
        enc_out = self.student.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_dict=True
        )

        props = batch["properties"].clamp(-10.0, 10.0)
        prop_emb = self.property_proj(props)
        prop_emb = prop_emb / prop_emb.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        prop_emb = PROPERTY_SCALE * prop_emb.unsqueeze(1)

        return enc_out.last_hidden_state + prop_emb

    # ==================================================
    # Training step
    # ==================================================
    def training_step(self, batch, batch_idx):
        # -------- teacher hidden states --------
        with torch.no_grad():
            t_h = self.teacher(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                properties=batch["properties"]
            )

        # -------- student hidden states --------
        s_h = self.forward_student_conditioned_encoder(batch)

        # -------- seq2seq loss --------
        enc_out = BaseModelOutput(last_hidden_state=s_h)
        s_out = self.student(
            encoder_outputs=enc_out,
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            return_dict=True
        )

        ce_loss = s_out.loss
        kd_loss = F.mse_loss(
            self.teacher_to_student_proj(t_h),
            s_h
        )

        loss = ce_loss + ALPHA * kd_loss

        # -------- logging (thesis-ready) --------
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/ce_loss", ce_loss, on_step=True, on_epoch=True)
        self.log("train/kd_loss", kd_loss, on_step=True, on_epoch=True)

        return loss
    # ==================================================
    # Validation step (for convergence monitoring)
    # ==================================================
    def validation_step(self, batch, batch_idx):
        s_h = self.forward_student_conditioned_encoder(batch)

        enc_out = BaseModelOutput(last_hidden_state=s_h)
        out = self.student(
            encoder_outputs=enc_out,
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            return_dict=True
        )

        val_loss = out.loss
        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        return val_loss

    # ==================================================
    # Optimizer
    # ==================================================

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-2)
    


def save_loss_plots(log_dir, outdir, tag):
    """
    Reads Lightning CSV logs and saves loss-vs-step plots.
    """
    metrics_path = os.path.join(log_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        print("❌ metrics.csv not found, skipping plots")
        return

    df = pd.read_csv(metrics_path)

    # -------------------------
    # Training loss vs step
    # -------------------------
    plt.figure(figsize=(6,4))
    if "train/loss_step" in df.columns:
        plt.plot(df["step"], df["train/loss_step"], label="Train Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title(f"Training Loss vs Step ({tag})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{tag}_train_loss.png"))
    plt.close()

    # -------------------------
    # Validation loss vs epoch
    # -------------------------
    if "val/loss" in df.columns:
        val_df = df.dropna(subset=["val/loss"])
        plt.figure(figsize=(6,4))
        plt.plot(val_df["epoch"], val_df["val/loss"], marker="o", label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Validation Loss vs Epoch ({tag})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{tag}_val_loss.png"))
        plt.close()

    # -------------------------
    # CE vs KD loss (optional but very nice)
    # -------------------------
    if "train/ce_loss_step" in df.columns and "train/kd_loss_step" in df.columns:
        plt.figure(figsize=(6,4))
        plt.plot(df["step"], df["train/ce_loss_step"], label="CE Loss")
        plt.plot(df["step"], df["train/kd_loss_step"], label="KD Loss")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title(f"CE vs KD Loss ({tag})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{tag}_ce_kd_loss.png"))
        plt.close()

    print(f"📈 Loss plots saved for {tag}")
    
# ============================================================
# MAIN
# ============================================================
def main():
    print("🔹 Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_PATH)

    dataset = BigSMILESDataset(
        csv_path=DATA_PATH,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    # ---- اضافه شده: validation split ----
    train_size = int((1 - VAL_RATIO) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    # ----------------------------------------

    os.makedirs(OUT_BASE, exist_ok=True)

    for scale in STUDENT_SCALES:
        tag = f"student {scale}"
        outdir = os.path.join(OUT_BASE, tag)
        model_dir = os.path.join(outdir, "model")
        os.makedirs(model_dir, exist_ok=True)

        print(f"\n🚀 Training {tag}")

        model = PolyTAOStudentPL(teacher_ckpt_path=TEACHER_PATH, capacity_percent=scale).to(DEVICE)

        ckpt_cb = ModelCheckpoint(
            dirpath=outdir,
            save_last=True,
            save_top_k=1,
            monitor="train/loss_epoch",
            mode="min"
        )

        csv_logger = CSVLogger(
            save_dir=outdir,
            name="logs"
        )

        trainer = Trainer(
            max_epochs=EPOCHS,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision="32",
            gradient_clip_val=1.0,
            callbacks=[ckpt_cb],
            logger=csv_logger,
            log_every_n_steps=50,
            enable_model_summary=False,
            val_check_interval=1.0
        )

        try:
            trainer.fit(model, train_loader, val_loader)
        finally:
            print(" Force-saving student model...")
            model.student.save_pretrained(model_dir, safe_serialization=True)
            tokenizer.save_pretrained(model_dir)

            torch.save(model.property_proj.state_dict(),
                    os.path.join(model_dir, "property_proj_student.pt"))
            torch.save(model.teacher_to_student_proj.state_dict(),
                    os.path.join(model_dir, "teacher_to_student_proj.pt"))

        # -------------------------
        # SAVE TRAINING PLOTS
        # -------------------------
        log_dir = csv_logger.log_dir
        save_loss_plots(
            log_dir=log_dir,
            outdir=outdir,
            tag=tag.replace(" ", "_")
        )

        # -------- SAVE --------
        model.student.generation_config.output_hidden_states = False
        model.student.generation_config.return_dict_in_generate = False

        model.student.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

        torch.save(model.property_proj.state_dict(),
                   os.path.join(model_dir, "property_proj_student.pt"))
        torch.save(model.teacher_to_student_proj.state_dict(),
                   os.path.join(model_dir, "teacher_to_student_proj.pt"))

        print(f"✅ {tag} saved to: {model_dir}")

    print("\n🎉 Student-50% training completed successfully.")


if __name__ == "__main__":
    main()
