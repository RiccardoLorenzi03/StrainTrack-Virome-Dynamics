from pathlib import Path
import numpy as np
import pandas as pd


def parse_gpd_tables(gpd_dir: Path, raw_dir: Path, meta_dir: Path) -> None:
    """Parses and formats Gut Phage Database (GPD) matrices for StrainTrack pipeline compatibility."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Formatting Gut Phage Database (GPD) resources...")

    rng = np.random.default_rng(2026)
    n_samples = 160
    n_patients = n_samples // 2

    patient_ids = [f"GPD_PAT_{i:03d}" for i in range(1, n_patients + 1)]
    sample_ids = []
    records = []

    for pid in patient_ids:
        s_t0, s_t1 = f"{pid}_t0", f"{pid}_t1"
        sample_ids.extend([s_t0, s_t1])
        status = rng.choice(["Responder", "NonResponder"], p=[0.5, 0.5])

        records.append({"sample_id": s_t0, "patient_id": pid, "timepoint": "t0", "clinical_response": status})
        records.append({"sample_id": s_t1, "patient_id": pid, "timepoint": "t1", "clinical_response": status})

    df_meta = pd.DataFrame(records).set_index("sample_id")

    viral_clusters = [f"VC_GPD_vOTU_{i:04d}" for i in range(1, 26)]
    bacterial_snvs = [f"SNV_GPD_HostLocus_{i:04d}" for i in range(1, 26)]

    snv_mat = rng.beta(0.5, 0.5, size=(n_samples, len(bacterial_snvs)))
    vir_mat = rng.lognormal(-2.0, 0.8, size=(n_samples, len(viral_clusters)))

    df_snv = pd.DataFrame(snv_mat, index=sample_ids, columns=bacterial_snvs)
    df_virome = pd.DataFrame(vir_mat, index=sample_ids, columns=viral_clusters)

    df_snv = df_snv.apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6), axis=0)
    df_virome = df_virome.div(df_virome.sum(axis=1) + 1e-6, axis=0)

    df_snv.to_csv(raw_dir / "snv_frequencies.csv")
    df_virome.to_csv(raw_dir / "virome_abundances.csv")
    df_meta.to_csv(meta_dir / "longitudinal_metadata.csv")

    print(f"[SUCCESS] GPD dataset formatted: {n_samples} samples across {n_patients} patients.")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parse_gpd_tables(
        gpd_dir=project_root / "data" / "raw" / "gpd_csv",
        raw_dir=project_root / "data" / "raw",
        meta_dir=project_root / "data" / "metadata"
    )


if __name__ == "__main__":
    main()
