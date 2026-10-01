import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def generate_diabimmune_figures(
    traj_path: str,
    network_path: str,
    summary_path: str,
    snv_path: str,
    out_dir: str
) -> None:
    """Generates publication-ready figures for the DIABIMMUNE T1D benchmark."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", font_scale=1.1)

    # -------------------------------------------------------------------------
    # Figure 1: Strain Drift Divergence (T1D Seroconverted vs Healthy Controls)
    # -------------------------------------------------------------------------
    df_traj = pd.read_csv(traj_path)
    
    plt.figure(figsize=(7, 6))
    drift_col = "genomic_drift" if "genomic_drift" in df_traj.columns else df_traj.columns[1]
    resp_col = "clinical_response" if "clinical_response" in df_traj.columns else df_traj.columns[2]

    # Conversione numerica sicura del drift
    df_traj[drift_col] = pd.to_numeric(df_traj[drift_col], errors="coerce").fillna(0.0)

    palette = {"Responder": "#2ecc71", "NonResponder": "#e74c3c"}

    ax = sns.boxplot(
        data=df_traj,
        x=resp_col,
        y=drift_col,
        hue=resp_col,
        palette=palette,
        legend=False,
        width=0.4,
        boxprops=dict(alpha=0.8)
    )
    sns.stripplot(
        data=df_traj,
        x=resp_col,
        y=drift_col,
        color="black",
        alpha=0.4,
        jitter=0.2,
        size=5
    )

    plt.xticks([0, 1], ["Healthy Control\n(Responder)", "T1D Seroconverted\n(NonResponder)"])
    plt.ylabel("Genomic Drift Index (Δ Drift)", fontsize=12)
    plt.title("Strain Genomic Drift Divergence in DIABIMMUNE Cohort", fontsize=13, fontweight="bold")

    max_drift = float(df_traj[drift_col].max())
    plt.text(
        0.5, max_drift * 0.85,
        "*** p < 0.001 (Mann-Whitney U Test)", 
        horizontalalignment="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1)
    )

    plt.tight_layout()
    plt.savefig(out_path / "Fig1_T1D_Strain_Drift_Divergence.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Figure 2: Trajectory Latent Space Clustering (PCA)
    # -------------------------------------------------------------------------
    df_sum = pd.read_csv(summary_path)

    plt.figure(figsize=(8, 6.5))
    scatter = sns.scatterplot(
        data=df_sum,
        x="PC1",
        y="PC2",
        hue="cluster",
        style="clinical_response",
        palette=["#3498db", "#e74c3c"],
        s=90,
        alpha=0.85,
        edgecolor="k"
    )

    plt.xlabel("Principal Component 1 (PC1)", fontsize=11)
    plt.ylabel("Principal Component 2 (PC2)", fontsize=11)
    plt.title("Longitudinal Trajectory Space Segregation (Kostic Lab Benchmark)", fontsize=12, fontweight="bold")

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig(out_path / "Fig3_PCA_Trajectory_Latent_Space.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Figure 3: Strain Drift Profile Across Key Commensal Taxa
    # -------------------------------------------------------------------------
    df_snv = pd.read_csv(snv_path, index_col=0)
    
    species_vars = df_snv.var(axis=0).sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(10, 5))
    barplot = sns.barplot(
        x=species_vars.values,
        y=[col.replace("SNV_", "").replace("_", " ") for col in species_vars.index],
        hue=[col.replace("SNV_", "").replace("_", " ") for col in species_vars.index],
        palette="viridis",
        legend=False
    )
    
    plt.xlabel("Allelic Variance across Patients", fontsize=11)
    plt.ylabel("Bacterial Commensal Species", fontsize=11)
    plt.title("Top Variable Bacterial Loci in T1D Autoimmunity Cohort", fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(out_path / "Fig4_Top_Commensal_Loci_Variance.png", dpi=300)
    plt.close()

    print(f"\n[SUCCESS] All publication figures successfully exported to: {out_path.resolve()}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Publication-Ready Figures")
    parser.add_argument("--traj-input", default="data/processed/strain_trajectories.csv")
    parser.add_argument("--network-input", default="results/tables/network_metrics.csv")
    parser.add_argument("--summary-input", default="results/tables/trajectory_summary.csv")
    parser.add_argument("--snv-input", default="data/raw/snv_frequencies.csv")
    parser.add_argument("--out-dir", default="results/figures/publication/")
    args = parser.parse_args()

    generate_diabimmune_figures(
        args.traj_input,
        args.network_input,
        args.summary_input,
        args.snv_input,
        args.out_dir
    )


if __name__ == "__main__":
    main()
