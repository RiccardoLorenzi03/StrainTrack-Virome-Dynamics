import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def compute_nucleotide_diversity(df_snv: pd.DataFrame) -> pd.Series:
    p = df_snv.values
    pi_per_site = 2.0 * p * (1.0 - p)
    return pd.Series(np.mean(pi_per_site, axis=1), index=df_snv.index, name="nucleotide_diversity")


def quantify_strain_dynamics(snv_path: str, meta_path: str, output_path: str) -> None:
    df_snv = pd.read_csv(snv_path, index_col=0)
    df_meta = pd.read_csv(meta_path, index_col=0)

    pi_series = compute_nucleotide_diversity(df_snv)
    df_meta["nucleotide_diversity"] = pi_series

    patients = df_meta["patient_id"].unique()
    trajectories = []

    for pid in patients:
        p_df = df_meta[df_meta["patient_id"] == pid]
        t0_samples = p_df[p_df["timepoint"] == "t0"].index
        t1_samples = p_df[p_df["timepoint"] == "t1"].index

        if len(t0_samples) > 0 and len(t1_samples) > 0:
            t0_id = t0_samples[0]
            t1_id = t1_samples[0]

            if t0_id in df_snv.index and t1_id in df_snv.index:
                dist = np.mean(np.abs(df_snv.loc[t0_id].values - df_snv.loc[t1_id].values))
                response = p_df["clinical_response"].iloc[0]
                
                event = "Strain_Replacement" if dist > 0.18 else "Strain_Persistence"
                
                trajectories.append({
                    "patient_id": pid,
                    "t0_sample_id": t0_id,
                    "t1_sample_id": t1_id,
                    "genomic_distance": dist,
                    "clinical_response": response,
                    "ecological_event": event
                })

    df_traj = pd.DataFrame(trajectories)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_traj.to_csv(output_path, index=False)

    if df_traj.empty:
        print("[WARNING] No matching paired t0/t1 samples found across metadata and SNV data.")
        return

    mean_pi = pi_series.mean()
    std_pi = pi_series.std()
    n_total = len(df_traj)
    n_replace = (df_traj["ecological_event"] == "Strain_Replacement").sum()
    n_persist = (df_traj["ecological_event"] == "Strain_Persistence").sum()
    pct_replace = (n_replace / n_total) * 100
    pct_persist = (n_persist / n_total) * 100

    resp_mask = df_traj["clinical_response"] == "Responder"
    nonresp_mask = df_traj["clinical_response"] == "NonResponder"

    dist_resp = df_traj[resp_mask]["genomic_distance"].mean() if resp_mask.any() else 0.0
    dist_nonresp = df_traj[nonresp_mask]["genomic_distance"].mean() if nonresp_mask.any() else 0.0

    print("\n" + "=" * 65)
    print(" [STAGE 2 STATISTICS] STRAIN GENOMIC DIVERSITY & DYNAMICS")
    print("=" * 65)
    print(f"  • Mean Nucleotide Diversity (π):   {mean_pi:.4f} ± {std_pi:.4f}")
    print(f"  • Total Patients Tracked:          {n_total}")
    print(f"  • Strain Persistence Events:      {n_persist} ({pct_persist:.1f}%)")
    print(f"  • Strain Replacement Events:      {n_replace} ({pct_replace:.1f}%)")
    print(f"  • Mean Drift (Responders):         {dist_resp:.4f}")
    print(f"  • Mean Drift (Non-Responders):     {dist_nonresp:.4f}")
    print("=" * 65 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snv-input", required=True)
    parser.add_argument("--meta-input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    quantify_strain_dynamics(args.snv_input, args.meta_input, args.output)


if __name__ == "__main__":
    main()
