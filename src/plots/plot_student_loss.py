import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\uni\Mastertheses\Workspace\PolyTao\poly_fold_rx\checkpoints\polytao_student_distilled\student-80\logs\version_0\metrics.csv")

plt.plot(df["step"], df["train/loss_step"], label="Total Loss")
plt.plot(df["step"], df["train/ce_loss_step"], label="CE Loss") 
plt.plot(df["step"], df["train/kd_loss_step"], label="KD Loss")

plt.xlabel("Training step")
plt.ylabel("Training Loss")
plt.ylim(0.0, 0.01)  
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
