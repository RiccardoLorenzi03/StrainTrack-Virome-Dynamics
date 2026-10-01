import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def select_dynamic_features(df: pd.DataFrame, max_features: int | None = None, var_quantile: float = 0.25) -> pd.DataFrame:
    """Dynamically selects informative features based on empirical variance distribution."""
    variances = df.var(axis=0)
    informative_vars = variances[variances > 1e-8]

    if informative_vars.empty:
        return df

    if max_features is not None and max_features > 0:
        selected_cols = informative_vars.sort_values(ascending=False).head(max_features).index
    else:
        # Selezione 100% dinamica: mantiene le feature con varianza superiore al quantile di rumore
        cutoff = informative_vars.quantile(var_quantile) if len(informative_vars) > 4 else 0.0
        selected_cols = informative_vars[informative_vars >= cutoff].index

    return df[selected_cols]


def build_bipartite_interaction_network(
    snv_path: str,
    virome_path: str,
    table_out: str,
    fig_out: str,
    max_features: int | None = None,
    corr_threshold: float = 0.25
) -> None:
    """Mines cross-domain interaction networks using adaptively selected genomic and viral features."""
    df_snv = pd.read_csv(snv_path, index_col=0)
    df_vir = pd.read_csv(virome_path, index_col=0)

    # Selezione dinamica e adattiva senza dimensioni prefissate
    snv_sub = select_dynamic_features(df_snv, max_features=max_features)
    vir_sub = select_dynamic_features(df_vir, max_features=max_features)

    n_snv = snv_sub.shape[1]
    n_vir = vir_sub.shape[1]

    corr_full = np.corrcoef(snv_sub.values.T, vir_sub.values.T)
    corr_matrix = pd.DataFrame(
        corr_full[:n_snv, n_snv:],
        index=snv_sub.columns,
        columns=vir_sub.columns
    )

    B = nx.Graph()
    snv_nodes = list(snv_sub.columns)
    vir_nodes = list(vir_sub.columns)

    B.add_nodes_from(snv_nodes, bipartite=0)
    B.add_nodes_from(vir_nodes, bipartite=1)

    edges_added = 0
    for snv in snv_nodes:
        for vc in vir_nodes:
            weight = corr_matrix.loc[snv, vc]
            if not np.isnan(weight) and abs(weight) >= corr_threshold:
                B.add_edge(snv, vc, weight=float(weight))
                edges_added += 1

    degree_centrality = nx.degree_centrality(B)
    metrics_df = pd.DataFrame([
        {"node": node, "degree_centrality": deg, "node_type": "Bacterial_SNV" if node in snv_nodes else "Viral_Cluster"}
        for node, deg in degree_centrality.items()
    ]).sort_values(by="degree_centrality", ascending=False)

    Path(table_out).parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(table_out, index=False)

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(B, seed=42)
    
    nx.draw_networkx_nodes(B, pos, nodelist=snv_nodes, node_color="skyblue", node_size=300, label="Host/Bacteria SNVs")
    nx.draw_networkx_nodes(B, pos, nodelist=vir_nodes, node_color="salmon", node_size=300, label="Viral Clusters (Phages)")
    nx.draw_networkx_edges(B, pos, alpha=0.5, edge_color="gray")
    nx.draw_networkx_labels(B, pos, font_size=7)

    plt.title("Phage-Host Bipartite Co-occurrence Network", fontsize=13)
    plt.legend(scatterpoints=1)
    plt.axis("off")
    
    Path(fig_out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_out, dpi=300)
    plt.close()

    density = nx.density(B)
    avg_degree = sum(dict(B.degree()).values()) / len(B.nodes) if len(B.nodes) > 0 else 0.0
    n_components = nx.number_connected_components(B)

    top_bact_df = metrics_df[metrics_df["node_type"] == "Bacterial_SNV"]
    top_vir_df = metrics_df[metrics_df["node_type"] == "Viral_Cluster"]

    top_bacterial_name = top_bact_df.iloc[0]["node"] if not top_bact_df.empty else "N/A"
    top_bacterial_dc = top_bact_df.iloc[0]["degree_centrality"] if not top_bact_df.empty else 0.0

    top_viral_name = top_vir_df.iloc[0]["node"] if not top_vir_df.empty else "N/A"
    top_viral_dc = top_vir_df.iloc[0]["degree_centrality"] if not top_vir_df.empty else 0.0

    print("\n" + "=" * 65)
    print(" [STAGE 3 ADVANCED STATISTICS] PHAGE-HOST NETWORK TOPOLOGY")
    print("=" * 65)
    print(f"  • Total Network Nodes:             {len(B.nodes)} ({n_snv} SNVs + {n_vir} VCs)")
    print(f"  • Inter-Domain Edges (|r| >= {corr_threshold}):  {edges_added}")
    print(f"  • Network Bipartite Density:       {density:.4f}")
    print(f"  • Average Node Degree:             {avg_degree:.2f}")
    print(f"  • Connected Components:            {n_components}")
    print(f"  • Top Bacterial SNV Hub:          {top_bacterial_name} (DC: {top_bacterial_dc:.3f})")
    print(f"  • Top Phage Viral Cluster Hub:     {top_viral_name} (DC: {top_viral_dc:.3f})")
    print("=" * 65 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mining Phage-Host Bipartite Interaction Networks")
    parser.add_argument("--snv-input", required=True)
    parser.add_argument("--virome-input", required=True)
    parser.add_argument("--table-out", required=True)
    parser.add_argument("--fig-out", required=True)
    parser.add_argument("--max-features", type=int, default=None, help="Optional hard limit on features; defaults to dynamic variance selection")
    parser.add_argument("--corr-threshold", type=float, default=0.25, help="Absolute correlation threshold for bipartite edge creation")
    args = parser.parse_args()

    build_bipartite_interaction_network(
        args.snv_input,
        args.virome_input,
        args.table_out,
        args.fig_out,
        max_features=args.max_features,
        corr_threshold=args.corr_threshold
    )


if __name__ == "__main__":
    main()
