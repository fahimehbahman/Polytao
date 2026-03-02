import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler


# -----------------------------
# CONFIG
# -----------------------------
INPUT_CSV = "notebook/PolyTAO-BigSMILES_Version-Latest/data/Bigsmiles_with_properties.csv"
OUTPUT_CSV = "notebook/PolyTAO-BigSMILES_Version-Latest/data/polymers_with_properties_normalized_train.csv"
SCALER_PATH = "notebook/PolyTAO-BigSMILES_Version-Latest/data/property_scaler.pkl"

PROPERTY_COLUMNS = [
    "MolWt",
    "RingCount",
    "NumAromaticRings",
    "NumHAcceptors",
    "NumHDonors",
    "NumHeteroatoms",
    "NumRotatableBonds",
]


def main():
    # 1. load data
    df = pd.read_csv(INPUT_CSV)

    # sanity check
    missing_cols = [c for c in PROPERTY_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"❌ Missing columns in CSV: {missing_cols}")

    # 2. drop rows with NaN properties (RDKit parse failed)
    prop_df = df[PROPERTY_COLUMNS]
    valid_mask = prop_df.notna().all(axis=1)

    df_valid = df[valid_mask].reset_index(drop=True)
    prop_df_valid = df_valid[PROPERTY_COLUMNS]

    print(f"✅ Valid rows: {len(df_valid)} / {len(df)}")

    # 3. normalize
    scaler = StandardScaler()
    normalized_props = scaler.fit_transform(prop_df_valid)

    normalized_df = pd.DataFrame(
        normalized_props,
        columns=[f"{c}_norm" for c in PROPERTY_COLUMNS]
    )

    # 4. merge back
    final_df = pd.concat([df_valid, normalized_df], axis=1)

    # 5. save outputs
    final_df.to_csv(OUTPUT_CSV, index=False)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print("Normalization complete")

if __name__ == "__main__":
    main()
