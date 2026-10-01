import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def cluster_longitudinal_trajectories(traj_path: str, virome_path: str, meta_path: str, table_out: str, fig_out: str) -> None:
    df_traj = pd.read_csv(traj_path)
    df_vir = pd.read_csv(virome_path, index_col=0)

    if df_traj.empty:
        print("[WARNING] Trajectory table is empty. Skipping clustering.")
        return

    valid_patients = []
    delta_vir_list = []
    genomic_dists = []

    for _, row in df_traj.iterrows():
        pid = row["patient_id"]
        t0_id = row["t0_sample_id"] if "t0_sample_id" in row and pd.notna(row["t0_sample_id"]) else f"{pid}_t0"
        t1_id = row["t1_sample_id"] if "t1_sample_id" in row and pd.notna(row["t1_sample_id"]) else f"{pid}_t1"
        
        if t0_id in df_vir.index and t1_id in df_vir.index:
            delta = df_vir.loc[t1_id].values - df_vir.loc[t0_id].values
            delta_vir_list.append(delta)
            valid_patients.append(pid)
            genomic_dists.append(row["genomic_distance"])

    if not delta_vir_list:
        print("[WARNING] No matching virome samples found for trajectories.")
        return

    df_valid_traj = df_traj[df_traj["patient_id"].isin(valid_patients)].copy()

    X_delta = pd.DataFrame(delta_vir_list, index=valid_patients)
    X_delta["genomic_distance"] = genomic_dists

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_delta.values)

    n_clusters = min(2, len(valid_patients))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pca)

    # Calcolo Silhouette Score per validare la qualità geometrica del clustering
    sil_score = silhouette_score(X_pca, clusters) if len(set(clusters)) > 1 else 0.0

    df_valid_traj["cluster"] = clusters
    df_valid_traj["PCA1"] = X_pca[:, 0]
    df_valid_traj["PCA2"] = X_pca[:, 1]

    Path(table_out).parent.mkdir(parents=True, exist_ok=True)
    df_valid_traj.to_csv(table_out, index=False)

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df_valid_traj, x="PCA1", y="PCA2", hue="clinical_response", style="ecological_event",
        s=100, palette="Set1"
    )
    plt.title("Unsupervised Longitudinal Trajectory Space (Strain Drift + Virome Shift)", fontsize=12)
    plt.xlabel("PCA Axis 1")
    plt.ylabel("PCA Axis 2")

    Path(fig_out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_out, dpi=300)
    plt.close()

    var_explained = pca.explained_variance_ratio_.sum() * 100
    cluster_counts = df_valid_traj["cluster"].value_counts().to_dict()
    ctable = pd.crosstab(df_valid_traj["cluster"], df_valid_traj["clinical_response"])

    c0_nonresp = ctable.loc[0, "NonResponder"] if "NonResponder" in ctable.columns and 0 in ctable.index else 0
    c0_total = cluster_counts.get(0, 1)
    c1_nonresp = ctable.loc[1, "NonResponder"] if "NonResponder" in ctable.columns and 1 in ctable.index else 0
    c1_total = cluster_counts.get(1, 1)

    print("\n" + "=" * 65)
    print(" [STAGE 4 ADVANCED STATISTICS] UNSUPERVISED TRAJECTORY CLUSTERING")
    print("=" * 65)
    print(f"  • PCA Variance Explained (2 PCs):  {var_explained:.1f}%")
    print(f"  • Silhouette Clustering Score:     {sil_score:.4f}")
    print(f"  • Cluster 0 Size:                  {cluster_counts.get(0, 0)} patients ({c0_nonresp/c0_total*100:.1f}% Non-Responders)")
    print(f"  • Cluster 1 Size:                  {cluster_counts.get(1, 0)} patients ({c1_nonresp/c1_total*100:.1f}% Non-Responders)")
    print("=" * 65 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-input", required=True)
    parser.add_argument("--virome-input", required=True)
    parser.add_argument("--meta-input", required=True)
    parser.add_argument("--table-out", required=True)
    parser.add_argument("--fig-out", required=True)
    args = parser.parse_args()

    cluster_longitudinal_trajectories(
        args.traj_input, args.virome_input, args.meta_input, args.table_out, args.fig_out
    )


if __name__ == "__main__":
    main()
