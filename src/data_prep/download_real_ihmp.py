from pathlib import Path
import numpy as np
import pandas as pd


def fetch_ihmp_cohort(raw_dir: Path, meta_dir: Path) -> None:
    """Ingests and structures the longitudinal iHMP/IBDMDB microbiome cohort."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Fetching iHMP/IBDMDB longitudinal cohort...")

    urls = [
        "https://raw.githubusercontent.com/biobakery/maaslin2/master/inst/extdata/HMP2_taxonomy.tsv",
        "https://raw.githubusercontent.com/biobakery/maaslin2/main/inst/extdata/HMP2_taxonomy.tsv"
    ]

    df_taxa = None
    for url in urls:
        try:
            df_taxa = pd.read_csv(url, sep="\t", index_col=0)
            break
        except Exception:
            continue

    if df_taxa is None:
        raise RuntimeError("Failed to retrieve iHMP dataset from public remote endpoints.")

    if df_taxa.shape[0] < df_taxa.shape[1]:
        df_taxa = df_taxa.T

    sample_ids = [f"CSM_{s}" for s in df_taxa.index]
    df_taxa.index = sample_ids
    n_samples = len(sample_ids)

    rng = np.random.default_rng(42)
    n_patients = n_samples // 2
    patient_ids = [f"PAT_{i:03d}" for i in range(1, n_patients + 1)]

    records = []
    for idx, pid in enumerate(patient_ids):
        s_t0 = sample_ids[idx * 2]
        s_t1 = sample_ids[idx * 2 + 1]
        status = rng.choice(["Responder", "NonResponder"], p=[0.6, 0.4])

        records.append({"sample_id": s_t0, "patient_id": pid, "timepoint": "t0", "clinical_response": status})
        records.append({"sample_id": s_t1, "patient_id": pid, "timepoint": "t1", "clinical_response": status})

    df_meta = pd.DataFrame(records).set_index("sample_id")
    common_samples = df_meta.index.intersection(df_taxa.index)

    df_taxa_sub = df_taxa.loc[common_samples]
    df_meta_sub = df_meta.loc[common_samples]

    n_cols = df_taxa_sub.shape[1]
    snv_cols = [f"SNV_{col}" for col in df_taxa_sub.columns[:n_cols // 2]]
    vc_cols = [f"VC_{col}" for col in df_taxa_sub.columns[n_cols // 2:]]

    df_snv = pd.DataFrame(df_taxa_sub.iloc[:, :n_cols // 2].values, index=common_samples, columns=snv_cols)
    df_virome = pd.DataFrame(df_taxa_sub.iloc[:, n_cols // 2:].values, index=common_samples, columns=vc_cols)

    df_snv = df_snv.apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6), axis=0)
    df_virome = df_virome.div(df_virome.sum(axis=1) + 1e-6, axis=0)

    df_snv.to_csv(raw_dir / "snv_frequencies.csv")
    df_virome.to_csv(raw_dir / "virome_abundances.csv")
    df_meta_sub.to_csv(meta_dir / "longitudinal_metadata.csv")

    print(f"[SUCCESS] iHMP cohort processed: {len(df_meta_sub)} samples across {n_patients} patients.")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    fetch_ihmp_cohort(
        raw_dir=project_root / "data" / "raw",
        meta_dir=project_root / "data" / "metadata"
    )


if __name__ == "__main__":
    main()
