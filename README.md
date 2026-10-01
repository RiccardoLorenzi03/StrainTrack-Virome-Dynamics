# StrainTrack-Virome-Dynamics

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Build](https://img.shields.io/badge/Pipeline-Bash%2FHPC-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

An integrated computational framework designed to track bacterial strain-level genomic micro-diversity (SNVs) and bacteriophage virome dynamics across longitudinal clinical cohorts.

---

## Key Methodological Features

1. **Strain-Level SNV & Allelic Diversity ($\pi$):** Calculates nucleotide diversity ($\pi$) and tracks strain replacement versus within-host genomic drift across longitudinal timepoints ($t_0 \to t_1$).
2. **Bipartite Phage-Host Graph Mining:** Models predator-prey interaction networks between Viral Clusters (VCs) and bacterial strain variants using graph centrality metrics (`NetworkX`).
3. **Unsupervised Longitudinal Trajectory Space:** Integrates multi-view features (microbial genomic drift + viral shifts) using PCA and $k$-means clustering to stratify clinical treatment outcomes.

---

## Quick Start

```bash
# Setup Conda environment
conda env create -f environment.yml
conda activate straintrack_env

# Execute full pipeline
bash pipeline/run_analysis.sh
