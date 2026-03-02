import os
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------
# Paths
# ------------------------
gen_df = pd.read_csv(
    r"C:\uni\Mastertheses\Workspace\PolyTao\poly_fold_rx\results\eval_student_100\valid_generated_molecules.csv"
)

test_df = pd.read_csv(
    r"C:\uni\Mastertheses\Workspace\PolyTao\poly_fold_rx\data\polymers_with_properties_normalized_test.csv"
)

OUT_DIR = r"C:\uni\Mastertheses\Workspace\PolyTao\poly_fold_rx\results\eval_student_100\histograms"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------
# Properties to plot
# ------------------------
PROPS = ["MolWt", "HeavyAtomCount", "RingCount"]

# ------------------------
# Plot & SAVE ONLY
# ------------------------
for p in PROPS:
    plt.figure(figsize=(5, 3))

    plt.hist(
        test_df[p],
        bins=30,
        density=True,
        alpha=0.6,
        label="Test"
    )

    plt.hist(
        gen_df[p],
        bins=30,
        density=True,
        alpha=0.6,
        label="Generated"
    )

    plt.xlabel(p)
    plt.ylabel("Density")
    plt.title(f"Generated vs Test — {p}")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, f"{p}_hist.png")
    plt.savefig(out_path, dpi=300)
    plt.close()  

print("✅ Histograms saved to:", OUT_DIR)
