import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def run_trajectory_clustering(
    traj_path: str,
    virome_path: str,
    meta_path: str,
    table_out: str,
    fig_out: str,
    n_clusters: int = 2
) -> None:
    """Performs unsupervised longitudinal trajectory clustering using PCA and k-means."""
    df_traj = pd.read_csv(traj_path, index_col=0)
    df_vir = pd.read_csv(virome_path, index_col=0)
    df_meta = pd.read_csv(meta_path, index_col=0)

    patient_features = []
    patient_ids = df_meta["patient_id"].unique()

    for pid in patient_ids:
        p_samples = df_meta[df_meta["patient_id"] == pid].index
        p_t0 = df_meta[(df_meta["patient_id"] == pid) & (df_meta["timepoint"] == "t0")].index
        p_t1 = df_meta[(df_meta["patient_id"] == pid) & (df_meta["timepoint"] == "t1")].index

        if len(p_t0) == 0 or len(p_t1) == 0:
            continue

        s_t0, s_t1 = p_t0[0], p_t1[0]

        # Drift dal modulo di diversità di ceppo
        drift_val = df_traj.loc[df_traj["patient_id"] == pid, "genomic_drift"].values
        drift = drift_val[0] if len(drift_val) > 0 else 0.0

        # Spostamento dinamico su TUTTE le feature del viroma
        v_t0 = df_vir.loc[s_t0].values
        v_t1 = df_vir.loc[s_t1].values
        v_shift = v_t1 - v_t0

        response = df_meta.loc[s_t0, "clinical_response"] if "clinical_response" in df_meta.columns else "Unknown"

        feat_vec = [drift] + list(v_shift)
        patient_features.append({"patient_id": pid, "clinical_response": response, "features": feat_vec})

    df_feat = pd.DataFrame(patient_features)
    X = np.array(df_feat["features"].tolist())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_explained = pca.explained_variance_ratio_.sum() * 100

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_pca)

    sil_score = silhouette_score(X_pca, cluster_labels) if len(np.unique(cluster_labels)) > 1 else 0.0

    df_feat["cluster"] = cluster_labels
    df_feat["PC1"] = X_pca[:, 0]
    df_feat["PC2"] = X_pca[:, 1]

    summary_df = df_feat[["patient_id", "clinical_response", "cluster", "PC1", "PC2"]]
    Path(table_out).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(table_out, index=False)

    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="viridis", alpha=0.8, edgecolors="k", s=60)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=11)
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=11)
    plt.title(f"Unsupervised Longitudinal Trajectory Space (Silhouette: {sil_score:.3f})", fontsize=12)
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
    print(f"  • PCA Variance Explained (2 PCs):  {var_explained:.1f}%")
    print(f"  • Silhouette Clustering Score:     {sil_score:.4f}")
    print(f"  • Cluster 0 Size:                  {len(c0)} patients ({c0_non_resp:.1f}% Non-Responders)")
    print(f"  • Cluster 1 Size:                  {len(c1)} patients ({c1_non_resp:.1f}% Non-Responders)")
    print("=" * 65 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unsupervised Trajectory Clustering")
    parser.add_argument("--traj-input", required=True)
    parser.add_argument("--virome-input", required=True)
    parser.add_argument("--meta-input", required=True)
    parser.add_argument("--table-out", required=True)
    parser.add_argument("--fig-out", required=True)
    args = parser.parse_args()

    run_trajectory_clustering(
        args.traj_input,
        args.virome_input,
        args.meta_input,
        args.table_out,
        args.fig_out
    )


if __name__ == "__main__":
    main()
