import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def detect_column(df: pd.DataFrame, candidates: list[str], fallback_type: str = "string") -> str:
    """Dynamically identifies target column based on naming heuristics or data type fallback."""
    for col in candidates:
        for c in df.columns:
            if col.lower() in str(c).lower():
                return c
    
    if fallback_type == "numeric":
        num_cols = df.select_dtypes(include=[np.number]).columns
        if not num_cols.empty:
            return num_cols[0]
    else:
        obj_cols = df.select_dtypes(include=["object", "category"]).columns
        if not obj_cols.empty:
            return obj_cols[0]

    return df.columns[0]


def run_trajectory_clustering(
    traj_path: str,
    virome_path: str,
    meta_path: str,
    table_out: str,
    fig_out: str,
    n_clusters: int = 2,
    target_variance: float = 0.80
) -> None:
    """Performs adaptive PCA trajectory clustering retaining target variance threshold before k-means."""
    df_traj = pd.read_csv(traj_path)
    df_vir = pd.read_csv(virome_path, index_col=0)
    df_meta = pd.read_csv(meta_path)

    if "Unnamed: 0" in df_meta.columns:
        df_meta = df_meta.rename(columns={"Unnamed: 0": "sample_id"}).set_index("sample_id")
    elif "sample_id" in df_meta.columns:
        df_meta = df_meta.set_index("sample_id")

    if "Unnamed: 0" in df_traj.columns:
        df_traj = df_traj.rename(columns={"Unnamed: 0": "index_id"})

    pid_col = detect_column(df_meta, ["patient_id", "subject_id", "host_id", "patient", "subject"])
    time_col = detect_column(df_meta, ["timepoint", "time", "visit", "tp"])
    resp_col = detect_column(df_meta, ["clinical_response", "response", "status", "group", "phenotype"])

    traj_pid_col = detect_column(df_traj, ["patient_id", "subject_id", "host_id", "patient", "subject"])
    traj_drift_col = detect_column(df_traj, ["genomic_drift", "drift", "distance", "pi_drift"], fallback_type="numeric")

    patient_features = []
    patient_ids = df_meta[pid_col].dropna().unique()

    for pid in patient_ids:
        p_meta = df_meta[df_meta[pid_col] == pid]
        
        timepoints = p_meta[time_col].dropna().unique()
        if len(timepoints) < 2:
            continue

        t0_val = "t0" if "t0" in timepoints else timepoints[0]
        t1_val = "t1" if "t1" in timepoints else timepoints[1]

        p_t0 = p_meta[p_meta[time_col] == t0_val]
        p_t1 = p_meta[p_meta[time_col] == t1_val]

        if p_t0.empty or p_t1.empty:
            continue

        s_t0, s_t1 = p_t0.index[0], p_t1.index[0]

        if s_t0 not in df_vir.index or s_t1 not in df_vir.index:
            continue

        p_traj = df_traj[df_traj[traj_pid_col] == pid]
        drift = 0.0
        if not p_traj.empty:
            val = pd.to_numeric(p_traj[traj_drift_col].values[0], errors="coerce")
            drift = float(val) if not np.isnan(val) else 0.0

        v_t0 = pd.to_numeric(df_vir.loc[s_t0], errors="coerce").fillna(0.0).values
        v_t1 = pd.to_numeric(df_vir.loc[s_t1], errors="coerce").fillna(0.0).values
        v_shift = v_t1 - v_t0

        response = str(p_t0[resp_col].values[0]) if resp_col in p_t0.columns else "Unknown"

        feat_vec = [drift] + list(v_shift)
        patient_features.append({"patient_id": pid, "clinical_response": response, "features": feat_vec})

    if not patient_features:
        raise RuntimeError("No valid longitudinal patient pairs found across input datasets.")

    df_feat = pd.DataFrame(patient_features)
    X = np.array(df_feat["features"].tolist())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Fit PCA su tutte le componenti possibili
    pca_full = PCA(random_state=42)
    X_pca_full = pca_full.fit_transform(X_scaled)

    # 2. Selezione dinamica delle PC per raggiungere il target di varianza (es. 80%)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components_needed = int(np.argmax(cum_var >= target_variance) + 1)
    
    # Se il numero di campioni/feature è molto piccolo, assicura almeno 2 componenti per il plot
    n_components_needed = max(2, min(n_components_needed, X_scaled.shape[1], X_scaled.shape[0]))

    # Space ridotto a K dimensioni per il clustering
    X_cluster = X_pca_full[:, :n_components_needed]
    var_explained_clustering = float(cum_var[n_components_needed - 1] * 100)

    # 3. Clustering k-means nello spazio k-dimensionale ottimale
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_cluster)

    sil_score = silhouette_score(X_cluster, cluster_labels) if len(np.unique(cluster_labels)) > 1 else 0.0

    df_feat["cluster"] = cluster_labels
    df_feat["PC1"] = X_pca_full[:, 0]
    df_feat["PC2"] = X_pca_full[:, 1]

    summary_df = df_feat[["patient_id", "clinical_response", "cluster", "PC1", "PC2"]]
    Path(table_out).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(table_out, index=False)

    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(X_pca_full[:, 0], X_pca_full[:, 1], c=cluster_labels, cmap="viridis", alpha=0.8, edgecolors="k", s=60)
    plt.xlabel(f"PC1 ({pca_full.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=11)
    plt.ylabel(f"PC2 ({pca_full.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=11)
    plt.title(f"PCA Latent Space ({n_components_needed} PCs explained {var_explained_clustering:.1f}% Var | Silhouette: {sil_score:.3f})", fontsize=11)
    plt.colorbar(scatter, label="Trajectory Cluster")

    Path(fig_out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_out, dpi=300)
    plt.close()

    c0 = summary_df[summary_df["cluster"] == 0]
    c1 = summary_df[summary_df["cluster"] == 1]

    c0_non_resp = (c0["clinical_response"] == "NonResponder").mean() * 100 if len(c0) > 0 else 0.0
    c1_non_resp = (c1["clinical_response"] == "NonResponder").mean() * 100 if len(c1) > 0 else 0.0

    print("\n" + "=" * 65)
    print(" [STAGE 4 ADVANCED STATISTICS] UNSUPERVISED TRAJECTORY CLUSTERING")
    print("=" * 65)
    print(f"  • Retained PCA Components:          {n_components_needed} PCs (explained {var_explained_clustering:.1f}% variance)")
    print(f"  • PC1 + PC2 2D Plot Variance:      {(pca_full.explained_variance_ratio_[0] + pca_full.explained_variance_ratio_[1])*100:.1f}%")
    print(f"  • High-Dim Silhouette Score:       {sil_score:.4f}")
    print(f"  • Cluster 0 Size:                  {len(c0)} patients ({c0_non_resp:.1f}% Non-Responders)")
    print(f"  • Cluster 1 Size:                  {len(c1)} patients ({c1_non_resp:.1f}% Non-Responders)")
    print("=" * 65 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive Multi-PC Unsupervised Trajectory Clustering")
    parser.add_argument("--traj-input", required=True)
    parser.add_argument("--virome-input", required=True)
    parser.add_argument("--meta-input", required=True)
    parser.add_argument("--table-out", required=True)
    parser.add_argument("--fig-out", required=True)
    parser.add_argument("--n-clusters", type=int, default=2)
    parser.add_argument("--target-variance", type=float, default=0.80, help="Target cumulative variance threshold for PCA dimension selection")
    args = parser.parse_args()

    run_trajectory_clustering(
        args.traj_input,
        args.virome_input,
        args.meta_input,
        args.table_out,
        args.fig_out,
        n_clusters=args.n_clusters,
        target_variance=args.target_variance
    )


if __name__ == "__main__":
    main()
