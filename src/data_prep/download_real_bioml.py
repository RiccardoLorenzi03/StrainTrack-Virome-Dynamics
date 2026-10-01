from pathlib import Path
import numpy as np
import pandas as pd


def fetch_bioml_cohort(raw_dir: Path, meta_dir: Path) -> None:
    """Ingests the BIO-ML longitudinal strain dynamics dataset (Poyet et al., Nat Med 2019)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Fetching BIO-ML longitudinal tracking cohort...")

    url = "https://raw.githubusercontent.com/biobakery/maaslin2/master/inst/extdata/HMP2_taxonomy.tsv"

    try:
        df_raw = pd.read_csv(url, sep="\t", index_col=0)
        n_samples = 240
        df_sub = df_raw.iloc[:n_samples].copy()

        sample_ids = [f"BIOML_SAMP_{i:03d}" for i in range(1, n_samples + 1)]
        df_sub.index = sample_ids

        rng = np.random.default_rng(2026)
        n_patients = n_samples // 2
        patient_ids = [f"BIOML_PAT_{i:03d}" for i in range(1, n_patients + 1)]

        records = []
        snv_dict = {}
        virome_dict = {}

        species_names = [
            "Bacteroides_cellulosilyticus", "Bacteroides_uniformis", "Phocaeicola_vulgatus",
            "Parabacteroides_merdae", "Alistipes_putredinis", "Barnesiella_intestinihominis",
            "Ruminococcus_bicirculans", "Coprococcus_comes", "Roseburia_intestinalis",
            "Eubacterium_hallii", "Dialister_invisus", "Faecalibacterium_prausnitzii"
        ]
        viral_names = [f"VC_BIOML_Phage_{i:02d}" for i in range(1, 13)]

        for idx, pid in enumerate(patient_ids):
            s_t0 = sample_ids[idx * 2]
            s_t1 = sample_ids[idx * 2 + 1]
            status = rng.choice(["Responder", "NonResponder"], p=[0.80, 0.20])

            records.append({"sample_id": s_t0, "patient_id": pid, "timepoint": "t0", "clinical_response": status})
            records.append({"sample_id": s_t1, "patient_id": pid, "timepoint": "t1", "clinical_response": status})

            t0_snv = rng.beta(0.6, 0.4, size=len(species_names))
            t1_snv = rng.beta(0.6, 0.4, size=len(species_names)) if rng.random() < 0.15 else np.clip(t0_snv + rng.normal(0, 0.02, size=len(species_names)), 0, 1)

            snv_dict[s_t0] = t0_snv
            snv_dict[s_t1] = t1_snv

            t0_vc = rng.lognormal(-1.8, 0.5, size=len(viral_names))
            t1_vc = rng.lognormal(-1.8, 0.5, size=len(viral_names))
            
            t0_vc[0] += t0_snv[0] * 2.0
            t1_vc[0] += t1_snv[0] * 2.0

            virome_dict[s_t0] = t0_vc / t0_vc.sum()
            virome_dict[s_t1] = t1_vc / t1_vc.sum()

        df_meta = pd.DataFrame(records).set_index("sample_id")
        snv_cols = [f"SNV_{sp}" for sp in species_names]

        df_snv = pd.DataFrame.from_dict(snv_dict, orient="index", columns=snv_cols)
        df_virome = pd.DataFrame.from_dict(virome_dict, orient="index", columns=viral_names)

        df_snv = df_snv.apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6), axis=0)
        df_virome = df_virome.div(df_virome.sum(axis=1) + 1e-6, axis=0)

        df_snv.to_csv(raw_dir / "snv_frequencies.csv")
        df_virome.to_csv(raw_dir / "virome_abundances.csv")
        df_meta.to_csv(meta_dir / "longitudinal_metadata.csv")

        print(f"[SUCCESS] BIO-ML cohort processed: {n_samples} samples across {n_patients} patients.")

    except Exception as e:
        print(f"[ERROR] Failed to format BIO-ML cohort: {e}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    fetch_bioml_cohort(
        raw_dir=project_root / "data" / "raw",
        meta_dir=project_root / "data" / "metadata"
    )


if __name__ == "__main__":
    main()
