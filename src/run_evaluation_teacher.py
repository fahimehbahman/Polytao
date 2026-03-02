# ============================================================
# run_evaluation_teacher.py
# Teacher Evaluation — Copy-Resistant (T5-safe)
# ============================================================

import argparse
import random
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from sklearn.metrics.pairwise import rbf_kernel

RDLogger.DisableLog("rdApp.error")

# ============================================================
# CONFIG
# ============================================================

SEED = 42
MAX_LEN = 256
PROPERTY_SCALE = 0.05
ENCODER_MASK_RATIO = 0.7   # 0.6–0.8 recommended

PROPERTY_BASE_COLS = [
    "MolWt", "HeavyAtomCount", "NHOHCount", "NOCount",
    "NumAliphaticCarbocycles", "NumAliphaticHeterocycles",
    "NumAliphaticRings", "NumAromaticCarbocycles",
    "NumAromaticHeterocycles", "NumAromaticRings",
    "NumHAcceptors", "NumHDonors", "NumHeteroatoms",
    "NumRotatableBonds", "RingCount",
]

# ============================================================
# UTILS
# ============================================================

def clean_smiles(s: str) -> str:
    if not s:
        return ""
    s = s.replace("<pad>", "").replace("</s>", "")
    return re.sub(r"\s+", "", s).strip()


def detect_polymer_column(df: pd.DataFrame) -> str:
    for c in ["product", "smiles", "SMILES", "polymer_smiles", "bigsmiles"]:
        if c in df.columns:
            return c
    raise ValueError(f"No SMILES column found. Available columns: {list(df.columns)}")


def mask_encoder_tokens(input_ids, tokenizer, mask_ratio: float):
    """
    T5-safe encoder corruption:
    randomly replace tokens with PAD token to prevent copying.
    """
    ids = input_ids.clone()
    special = {
        tokenizer.pad_token_id,
        tokenizer.eos_token_id,
        tokenizer.bos_token_id,
    }

    for i in range(ids.size(1)):
        tok = ids[0, i].item()
        if tok in special:
            continue
        if random.random() < mask_ratio:
            ids[0, i] = tokenizer.pad_token_id
    return ids


def is_valid_smiles(smiles: str) -> bool:
    s = clean_smiles(smiles)
    if s == "":
        return False
    try:
        mol = Chem.MolFromSmiles(s, sanitize=False)
        if mol is None:
            return False
        Chem.SanitizeMol(mol)
        return True
    except Exception:
        return False


def rdkit_properties(smiles: str):
    if not is_valid_smiles(smiles):
        return None
    mol = Chem.MolFromSmiles(clean_smiles(smiles))
    return {p: getattr(Descriptors, p)(mol) for p in PROPERTY_BASE_COLS}


def compute_mmd(x, y, gamma=None):
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).reshape(-1, 1)
    if gamma is None:
        gamma = 1.0 / (np.std(np.vstack([x, y])) + 1e-8)
    return float(
        rbf_kernel(x, x, gamma).mean()
        + rbf_kernel(y, y, gamma).mean()
        - 2 * rbf_kernel(x, y, gamma).mean()
    )

# ============================================================
# MAIN
# ============================================================

def main(args):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = Path(args.model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    model.eval()

    # ---- property projection ----
    proj = nn.Linear(len(PROPERTY_BASE_COLS), model.config.d_model).to(device)
    proj.load_state_dict(torch.load(model_dir / "property_proj.pt", map_location=device))
    proj.eval()

    # ---- load dataset ----
    df = pd.read_csv(args.csv_path)
    polymer_col = detect_polymer_column(df)

    df = df.sample(min(args.n_generate, len(df)), random_state=SEED).reset_index(drop=True)

    prompts = df[polymer_col].astype(str).tolist()
    targets = np.stack([df[p].values for p in PROPERTY_BASE_COLS], axis=1).astype(np.float32)

    # shuffle prompts to break identity mapping
    rng = np.random.default_rng(SEED)
    rng.shuffle(prompts)

    generated = []

    with torch.no_grad():
        for p, t in tqdm(
            zip(prompts, targets),
            total=len(targets),
            desc="Generating (masked encoder)"
        ):
            enc = tokenizer(p, return_tensors="pt", truncation=True).to(device)

            # 🔥 prevent copying (T5-safe)
            enc["input_ids"] = mask_encoder_tokens(
                enc["input_ids"], tokenizer, ENCODER_MASK_RATIO
            )

            enc_out = model.encoder(**enc, return_dict=True)

            props = torch.tensor(t).unsqueeze(0).to(device)
            prop_emb = proj(props)
            prop_emb = prop_emb / prop_emb.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            prop_emb = PROPERTY_SCALE * prop_emb.unsqueeze(1)

            encoder_outputs = BaseModelOutput(
                last_hidden_state=enc_out.last_hidden_state + prop_emb
            )

            gen_ids = model.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=enc["attention_mask"],
                max_length=MAX_LEN,
                do_sample=False,
                num_beams=1,
            )

            generated.append(tokenizer.decode(gen_ids[0], skip_special_tokens=True))

    # ========================================================
    # EVALUATION
    # ========================================================

    rows = []
    for i, g in enumerate(generated):
        props = rdkit_properties(g)
        if props is None:
            continue
        row = {"generated": clean_smiles(g)}
        row.update(props)
        row.update({f"{p}_target": float(targets[i, j]) for j, p in enumerate(PROPERTY_BASE_COLS)})
        rows.append(row)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    valid_df = pd.DataFrame(rows)
    valid_df.to_csv(out / "teacher_valid_samples.csv", index=False)

    validity = len(valid_df) / max(1, len(generated))

    copy_rate = np.mean([
        clean_smiles(g) == clean_smiles(p)
        for g, p in zip(generated, prompts)
    ])

    with open(out / "validity.json", "w") as f:
        json.dump({"validity": validity, "n_generated": len(generated), "n_valid": len(valid_df)}, f, indent=2)

    with open(out / "copy_rate.json", "w") as f:
        json.dump({"copy_rate": float(copy_rate)}, f, indent=2)

    print(f"✅ Validity: {validity:.3f} ({len(valid_df)}/{len(generated)})")
    print(f"📄 Copy-rate: {copy_rate:.3f}")

    if len(valid_df) == 0:
        print("❌ No valid molecules — skipping metrics.")
        return

    metrics = []
    for p in PROPERTY_BASE_COLS:
        g = valid_df[p].values
        t = valid_df[f"{p}_target"].values
        metrics.append({
            "property": p,
            "mae": float(np.mean(np.abs(g - t))),
            "rmse": float(np.sqrt(np.mean((g - t) ** 2))),
            "mmd": compute_mmd(g, t),
            "corr": float(np.corrcoef(g, t)[0, 1]) if np.std(g) > 0 else np.nan,
            "n_valid": len(valid_df),
            "n_generated": len(generated),
        })

    pd.DataFrame(metrics).to_csv(out / "metrics_summary.csv", index=False)
    print("💾 Metrics saved.")

# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_generate", type=int, default=500)
    main(parser.parse_args())
