# ============================================================
# run_evaluation.py
# Unified Evaluation for Teacher & Student Models
# FINAL — CI-safe, Docker-safe, César-safe
# ============================================================

import argparse
import random
import re
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
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
    s = re.sub(r"\s+", "", s)
    return s.strip()


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


def rdkit_properties_raw(smiles: str):
    if not is_valid_smiles(smiles):
        return None
    mol = Chem.MolFromSmiles(clean_smiles(smiles))
    return {p: getattr(Descriptors, p)(mol) for p in PROPERTY_BASE_COLS}

# ============================================================
# MMD
# ============================================================

def compute_mmd(x, y, gamma=None):
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    if gamma is None:
        gamma = 1.0 / (np.std(np.vstack([x, y])) + 1e-8)

    k_xx = rbf_kernel(x, x, gamma=gamma)
    k_yy = rbf_kernel(y, y, gamma=gamma)
    k_xy = rbf_kernel(x, y, gamma=gamma)

    return float(k_xx.mean() + k_yy.mean() - 2 * k_xy.mean())

# ============================================================
# NORMALIZATION
# ============================================================

def build_targets_and_normalizers(df):
    targets, norm_info = [], []

    for p in PROPERTY_BASE_COLS:
        if f"{p}_norm" in df.columns:
            targets.append(df[f"{p}_norm"].values)
            norm_info.append({
                "prop": p,
                "use_norm": True,
                "mean": df[p].mean(),
                "std": df[p].std(ddof=0) + 1e-8,
            })
        elif p in df.columns:
            targets.append(df[p].values)
            norm_info.append({
                "prop": p,
                "use_norm": False,
            })
        else:
            raise ValueError(f"Missing property column: {p}")

    return np.stack(targets, axis=1).astype(np.float32), norm_info


def convert_generated_raw_to_condition_space(gen_raw_df, norm_info):
    out = {}
    for info in norm_info:
        p = info["prop"]
        if info.get("use_norm", False):
            out[p] = (gen_raw_df[p] - info["mean"]) / info["std"]
        else:
            out[p] = gen_raw_df[p]
    return pd.DataFrame(out)

# ============================================================
# GENERATION
# ============================================================

def generate_conditioned(model, tokenizer, proj, prompts, targets, device, args):
    outputs = []
    model.eval()

    with torch.no_grad():
        for p, t in tqdm(zip(prompts, targets), total=len(prompts)):
            enc = tokenizer(p, return_tensors="pt", truncation=True).to(device)
            enc_out = model.encoder(**enc, return_dict=True)

            props = torch.tensor(t, dtype=torch.float32).unsqueeze(0).to(device)
            props = props.clamp(-10.0, 10.0)

            prop_emb = proj(props)
            prop_emb = prop_emb / prop_emb.norm(dim=-1, keepdim=True)
            prop_emb = args.property_scale * prop_emb.unsqueeze(1)

            encoder_outputs = BaseModelOutput(
                last_hidden_state=enc_out.last_hidden_state + prop_emb
            )

            gen_ids = model.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=enc["attention_mask"],
                max_length=MAX_LEN,
                do_sample=True,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                num_beams=args.num_beams,
                use_cache=False,
            )

            outputs.append(
                tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            )

    return outputs

# ============================================================
# MAIN
# ============================================================

def main(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = out_dir / "metrics_summary.csv"
    validity_file = out_dir / "validity.json"

    if metrics_file.exists() and validity_file.exists() and not args.force:
        print("✅ Cached evaluation found — skipping recomputation.")
        return

    tokenizer = AutoTokenizer.from_pretrained("hkqiu/PolyTAO-BigSMILES_Version")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(device)

    proj = nn.Linear(15, model.config.d_model).to(device)
    proj_file = "property_proj.pt" if args.is_teacher else "property_proj_student.pt"
    proj.load_state_dict(torch.load(Path(args.model_path) / proj_file, map_location=device))
    proj.eval()

    df = pd.read_csv(args.train_csv)

    df_eval = df.sample(min(args.n_generate, len(df)), random_state=args.seed)
    prompts = df_eval["product"].tolist()

    
    if args.use_cli_properties:
        if args.properties is None:
            raise ValueError("❌ --use_cli_properties requires --properties")

        prop_vec = np.array(args.properties, dtype=np.float32)
        targets_eval = np.repeat(prop_vec[None, :], len(prompts), axis=0)
        norm_info = [{"prop": p, "use_norm": False} for p in PROPERTY_BASE_COLS]
    else:
        targets_eval, norm_info = build_targets_and_normalizers(df_eval)

    generated = generate_conditioned(
        model, tokenizer, proj, prompts, targets_eval, device, args
    )

    rows = []
    for i, g in enumerate(generated):
        props = rdkit_properties_raw(g)
        if props is None:
            continue
        row = {"generated": g} | props
        row.update({
            f"{p}_target": targets_eval[i, j]
            for j, p in enumerate(PROPERTY_BASE_COLS)
        })
        rows.append(row)

    valid_df = pd.DataFrame(rows)
    n_valid = len(valid_df)
    n_total = len(generated)

    valid_df.to_csv(out_dir / "valid_generated_molecules.csv", index=False)

    with open(validity_file, "w") as f:
        json.dump({
            "n_generated": n_total,
            "n_valid": n_valid,
            "validity_rate": n_valid / max(1, n_total)
        }, f, indent=2)

    if valid_df.empty:
        print("❌ No valid molecules — aborting metrics.")
        return

    gen_cond = convert_generated_raw_to_condition_space(valid_df, norm_info)

    summary = []
    for p in PROPERTY_BASE_COLS:
        t = valid_df[f"{p}_target"].values
        g = gen_cond[p].values
        summary.append({
            "property": p,
            "mae": np.mean(np.abs(t - g)),
            "rmse": np.sqrt(np.mean((t - g) ** 2)),
            "mmd": compute_mmd(t, g),
            "corr": np.corrcoef(t, g)[0, 1] if np.std(t) > 0 else np.nan,
        })

    pd.DataFrame(summary).to_csv(metrics_file, index=False)
    print("✅ Evaluation completed.")

# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--n_generate", type=int, default=500)
    parser.add_argument("--property_scale", type=float, default=0.05)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)

    parser.add_argument("--is_teacher", action="store_true")
    parser.add_argument("--force", action="store_true")

    # 🔴 فقط این دو تا جدید
    parser.add_argument("--use_cli_properties", action="store_true")
    parser.add_argument("--properties", type=float, nargs=15)

    main(parser.parse_args())
