#!/usr/bin/env python3
"""Plot results from the energy transformer test run."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "energy_transformer_test"

df = pd.read_csv(RESULTS_DIR / "test_results.csv")
hist = pd.read_csv(RESULTS_DIR / "training_history.csv")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Energy Transformer — Test Results (10k events, 5 epochs)", fontsize=14)

# --- 1. Energy: pred vs true ---
ax = axes[0, 0]
ax.scatter(df["log10E_true"], df["log10E_pred"], s=3, alpha=0.3)
lims = [df["log10E_true"].min(), df["log10E_true"].max()]
ax.plot(lims, lims, "r--", lw=1)
ax.set_xlabel("log10(E_true / GeV)")
ax.set_ylabel("log10(E_pred / GeV)")
ax.set_title("Energy: pred vs true")

# --- 2. Energy residual distribution ---
ax = axes[0, 1]
res = df["log10E_residual"]
ax.hist(res, bins=80, edgecolor="black", linewidth=0.3)
ax.axvline(0, color="r", ls="--", lw=1)
ax.set_xlabel("log10(E_pred) - log10(E_true)")
ax.set_ylabel("Count")
ax.set_title(f"Residuals (std={res.std():.3f})")

# --- 3. Relative error vs true energy ---
ax = axes[0, 2]
ax.scatter(df["log10E_true"], df["rel_error"], s=3, alpha=0.3)
ax.set_xlabel("log10(E_true / GeV)")
ax.set_ylabel("Relative error |E_pred - E_true| / E_true")
ax.set_ylim(0, 2)
ax.axhline(y=np.median(df["rel_error"]), color="r", ls="--", lw=1,
           label=f"median = {np.median(df['rel_error']):.3f}")
ax.legend()
ax.set_title("Relative error vs energy")

# --- 4. Stopped score distribution ---
ax = axes[1, 0]
has_labels = df["stopped_label"] >= 0
if has_labels.any():
    through = df.loc[(df["stopped_label"] == 0), "stopped_score"]
    stopped = df.loc[(df["stopped_label"] == 1), "stopped_score"]
    bins = np.linspace(0, 1, 51)
    ax.hist(through, bins=bins, alpha=0.6, label=f"Through ({len(through)})", density=True)
    ax.hist(stopped, bins=bins, alpha=0.6, label=f"Stopped ({len(stopped)})", density=True)
    ax.legend()
ax.set_xlabel("Stopped score σ(logit)")
ax.set_ylabel("Density")
ax.set_title(f"Stopped/through (acc={df['stopped_score'].notna().sum()})")

# --- 5. Position-z: pred vs true (stopped only) ---
ax = axes[1, 1]
valid = df["pos_z_valid"] == True
if valid.any():
    dv = df[valid]
    ax.scatter(dv["pos_z_true_m"], dv["pos_z_pred_m"], s=5, alpha=0.4)
    lims = [dv["pos_z_true_m"].min(), dv["pos_z_true_m"].max()]
    ax.plot(lims, lims, "r--", lw=1)
    mae = np.abs(dv["pos_z_pred_m"] - dv["pos_z_true_m"]).mean()
    ax.set_title(f"Pos_z stopped only (MAE={mae:.0f} m)")
else:
    ax.set_title("Pos_z: no valid labels")
ax.set_xlabel("pos_z true (m)")
ax.set_ylabel("pos_z pred (m)")

# --- 6. Training curves ---
ax = axes[1, 2]
ax.plot(hist["epoch"], hist["train_energy_loss"], label="train energy loss")
ax.plot(hist["epoch"], hist["val_mse_log10"], label="val MSE(log10 E)")
if "train_stop_loss" in hist.columns:
    ax.plot(hist["epoch"], hist["train_stop_loss"], label="train stop loss", ls="--")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend(fontsize=8)
ax.set_title("Training curves")

plt.tight_layout()
out = RESULTS_DIR / "results_overview.png"
plt.savefig(out, dpi=150)
print(f"Saved: {out}")

# --- Figure 2: Distribution-style plots (like the reference) ---
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle("Energy Transformer — Distribution Plots", fontsize=14)

# 2a. Energy distribution: truth vs pred (overlaid)
ax = axes2[0]
bins_e = np.linspace(0, 2000, 80)
ax.hist(df["energy_true_GeV"], bins=bins_e, alpha=0.6, density=True, label="Truth")
ax.hist(df["energy_pred_GeV"], bins=bins_e, alpha=0.6, density=True, label="Pred")
ax.set_xlabel("Energy [GeV]")
ax.set_ylabel("Density")
ax.set_title("Energy: Truth vs Pred")
ax.legend()

# 2b. Absolute error distribution
ax = axes2[1]
abs_err = np.abs(df["energy_pred_GeV"] - df["energy_true_GeV"])
mae = abs_err.mean()
median_err = np.median(abs_err)
q68 = np.quantile(abs_err, 0.68)
ax.hist(abs_err, bins=100, range=(0, 1000), edgecolor="black", linewidth=0.3)
ax.axvline(median_err, color="r", ls="--", lw=1, label=f"Median = {median_err:.1f} GeV")
ax.axvline(mae, color="k", ls="--", lw=1, label=f"Mean = {mae:.1f} GeV")
ax.axvline(q68, color="orange", ls="--", lw=1, label=f"q68 = {q68:.1f} GeV")
ax.set_xlabel("Absolute error [GeV]")
ax.set_ylabel("Count")
ax.set_title("Energy absolute error")
ax.legend(fontsize=8)

# 2c. Residual (pred - true) distribution
ax = axes2[2]
residual = df["energy_pred_GeV"] - df["energy_true_GeV"]
rmean = residual.mean()
rstd = residual.std()
ax.hist(residual, bins=100, range=(-1000, 500), edgecolor="black", linewidth=0.3,
        color="salmon")
ax.axvline(0, color="k", ls="--", lw=1)
ax.set_xlabel("Residual [GeV]")
ax.set_ylabel("Count")
ax.set_title(f"Energy residual (pred - true)\nmean={rmean:.1f}, std={rstd:.1f} GeV")

plt.tight_layout()
out2 = RESULTS_DIR / "results_distributions.png"
plt.savefig(out2, dpi=150)
print(f"Saved: {out2}")
plt.show()
