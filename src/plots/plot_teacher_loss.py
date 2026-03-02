import pandas as pd
import matplotlib.pyplot as plt

csv_path = r"C:\uni\Mastertheses\Workspace\PolyTao\poly_fold_rx\src\lightning_logs\version_0\teacher_training_loss.csv"

df = pd.read_csv(csv_path)

loss_df = df[["step", "loss"]].dropna()

plt.figure(figsize=(7, 4))
plt.plot(loss_df["step"], loss_df["loss"], linewidth=2)

plt.xlabel("Training Step")
plt.ylabel("Training Loss")
plt.title("Teacher Model Training Loss")
plt.grid(True)
plt.tight_layout()

plt.savefig("teacher_training_loss.png", dpi=300)
plt.show()
