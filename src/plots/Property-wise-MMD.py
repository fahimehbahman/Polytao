import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# path to CSV log
log_path = Path(
    r"C:\uni\Mastertheses\Workspace\PolyTao\poly_fold_rx\checkpoints"
    r"\polytao_student_distilled\student_40\logs\version_0\metrics.csv"
)

# load metrics
df = pd.read_csv(log_path)

# keep only training loss rows
loss_df = df[["step", "train/loss_step"]].dropna()

# plot
plt.figure(figsize=(7, 5))
plt.plot(
    loss_df["step"],
    loss_df["train/loss_step"],
    linewidth=2
)

plt.xlabel("Training Step")
plt.ylabel("Training Loss")
plt.title("Training Loss Curve — Student 40%")
plt.grid(True)
plt.tight_layout()
plt.show()
