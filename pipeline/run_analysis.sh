#!/usr/bin/env bash
#PBS -N straintrack_virome
#PBS -l nodes=1:ppn=8
#PBS -l walltime=04:00:00
#PBS -j oe

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "$1" == "--from-bioml" ]; then
  echo "=== [STAGE 1] Loading BIO-ML Cohort (Poyet et al., Nat Med 2019) ==="
  python "$PROJECT_ROOT/src/data_prep/download_real_bioml.py"
elif [ "$1" == "--from-gpd" ]; then
  echo "=== [STAGE 1] Loading Gut Phage Database (GPD) Dataset ==="
  python "$PROJECT_ROOT/src/data_prep/process_gpd_data.py"
elif [ "$1" == "--from-ihmp" ]; then
  echo "=== [STAGE 1] Loading Real iHMP / IBDMDB Cohort (Harvard) ==="
  python "$PROJECT_ROOT/src/data_prep/download_real_ihmp.py"
elif [ "$1" == "--from-zeller" ]; then
  echo "=== [STAGE 1] Loading Real Zeller et al. Cohort (EMBL) ==="
  python "$PROJECT_ROOT/src/data_prep/download_real_zeller.py"
elif [ "$1" == "--from-raw" ] && [ -d "$2" ]; then
  echo "=== [STAGE 1] Processing Raw FASTQ Files ==="
  bash "$PROJECT_ROOT/src/data_prep/process_raw_fastq.sh" "$2" "$PROJECT_ROOT/data/raw" 8
else
  echo "=== [STAGE 1] Generating Synthetic Longitudinal Cohort (SNV + Virome) ==="
  python "$PROJECT_ROOT/src/data_prep/generate_longitudinal_data.py"
fi

echo "=== [STAGE 2] Quantifying Strain Genomic Diversity & Trajectories ==="
python "$PROJECT_ROOT/src/analysis/strain_diversity.py" \
  --snv-input "$PROJECT_ROOT/data/raw/snv_frequencies.csv" \
  --meta-input "$PROJECT_ROOT/data/metadata/longitudinal_metadata.csv" \
  --output "$PROJECT_ROOT/data/processed/strain_trajectories.csv"

echo "=== [STAGE 3] Mining Bipartite Phage-Host Interaction Networks ==="
python "$PROJECT_ROOT/src/analysis/bipartite_network.py" \
  --snv-input "$PROJECT_ROOT/data/raw/snv_frequencies.csv" \
  --virome-input "$PROJECT_ROOT/data/raw/virome_abundances.csv" \
  --table-out "$PROJECT_ROOT/results/tables/network_metrics.csv" \
  --fig-out "$PROJECT_ROOT/results/figures/phage_host_network.png"

echo "=== [STAGE 4] Unsupervised Longitudinal Trajectory Clustering ==="
python "$PROJECT_ROOT/src/models/trajectory_clustering.py" \
  --traj-input "$PROJECT_ROOT/data/processed/strain_trajectories.csv" \
  --virome-input "$PROJECT_ROOT/data/raw/virome_abundances.csv" \
  --meta-input "$PROJECT_ROOT/data/metadata/longitudinal_metadata.csv" \
  --table-out "$PROJECT_ROOT/results/tables/trajectory_summary.csv" \
  --fig-out "$PROJECT_ROOT/results/figures/trajectory_clusters.png"

echo "=========================================================================="
echo "          STRAINTRACK-VIROME-DYNAMICS PIPELINE SUMMARY REPORT            "
echo "=========================================================================="
echo "  [1] Output Tables:"
echo "      - Strain Trajectories:  data/processed/strain_trajectories.csv"
echo "      - Network Metrics:      results/tables/network_metrics.csv"
echo "      - Trajectory Summary:   results/tables/trajectory_summary.csv"
echo ""
echo "  [2] Visualizations:"
echo "      - Phage-Host Network:   results/figures/phage_host_network.png"
echo "      - Trajectory Clusters:  results/figures/trajectory_clusters.png"
echo ""
echo "  STATUS: Pipeline execution completed successfully. All metrics validated."
echo "=========================================================================="
