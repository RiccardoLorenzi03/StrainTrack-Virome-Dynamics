#!/usr/bin/env bash
set -e

SOURCE_FLAG="${1:---from-ihmp}"

echo "=========================================================================="
echo "      STRAINTRACK-VIROME-DYNAMICS PIPELINE (HARVARD / KOSTIC LAB)         "
echo "=========================================================================="

if [ "$SOURCE_FLAG" == "--from-ihmp" ]; then
    echo "=== [STAGE 1] Loading Real iHMP / IBDMDB Cohort (Harvard) ==="
    python src/data_prep/download_real_ihmp.py
elif [ "$SOURCE_FLAG" == "--from-gpd" ]; then
    echo "=== [STAGE 1] Loading Gut Phage Database (GPD) Dataset ==="
    python src/data_prep/process_gpd_data.py
elif [ "$SOURCE_FLAG" == "--from-bioml" ]; then
    echo "=== [STAGE 1] Loading BIO-ML Cohort (Poyet et al., Nat Med 2019) ==="
    python src/data_prep/download_real_bioml.py
elif [ "$SOURCE_FLAG" == "--from-diabimmune" ]; then
    echo "=== [STAGE 1] Loading DIABIMMUNE T1D Cohort (Kostic et al., 2015) ==="
    python src/data_prep/download_real_diabimmune.py
else
    echo "[ERROR] Unknown flag '$SOURCE_FLAG'. Options: --from-ihmp, --from-gpd, --from-bioml, --from-diabimmune"
    exit 1
fi

echo "=== [STAGE 2] Quantifying Strain Genomic Diversity & Trajectories ==="
python src/analysis/strain_diversity.py \
    --snv-input data/raw/snv_frequencies.csv \
    --meta-input data/metadata/longitudinal_metadata.csv \
    --output data/processed/strain_trajectories.csv

echo "=== [STAGE 3] Mining Bipartite Phage-Host Interaction Networks ==="
python src/analysis/bipartite_network.py \
    --snv-input data/raw/snv_frequencies.csv \
    --virome-input data/raw/virome_abundances.csv \
    --table-out results/tables/network_metrics.csv \
    --fig-out results/figures/phage_host_network.png

echo "=== [STAGE 4] Unsupervised Longitudinal Trajectory Clustering ==="
python src/models/trajectory_clustering.py \
    --traj-input data/processed/strain_trajectories.csv \
    --virome-input data/raw/virome_abundances.csv \
    --meta-input data/metadata/longitudinal_metadata.csv \
    --table-out results/tables/trajectory_summary.csv \
    --fig-out results/figures/trajectory_clusters.png

echo "=== [STAGE 5] Exporting Publication-Grade Figures ==="
python src/visualization/generate_publication_figures.py \
    --traj-input data/processed/strain_trajectories.csv \
    --network-input results/tables/network_metrics.csv \
    --summary-input results/tables/trajectory_summary.csv \
    --snv-input data/raw/snv_frequencies.csv \
    --out-dir results/figures/publication/

echo "=========================================================================="
echo " STATUS: Pipeline execution completed successfully. All figures saved.   "
echo "=========================================================================="
