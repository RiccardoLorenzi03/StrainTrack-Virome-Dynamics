from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationConfig:
    n_patients: int = 60
    n_snv_loci: int = 50
    n_viral_clusters: int = 30
    seed: int = 2026


def generate_longitudinal_dataset(config: SimulationConfig, raw_dir: Path, meta_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.seed)
    
    patient_ids = [f"PATIENT_{i:03d}" for i in range(1, config.n_patients + 1)]
    meta_records = []
    
    snv_cols = [f"SNV_Locus_{i:03d}" for i in range(1, config.n_snv_loci + 1)]
    vc_cols = [f"VC_Cluster_{i:03d}" for i in range(1, config.n_viral_clusters + 1)]

    snv_dict = {}
    virome_dict = {}

    for pid in patient_ids:
        responder = rng.choice(["Responder", "NonResponder"], p=[0.5, 0.5])
        
        # 1. Baseline t0 SNVs
        t0_snv = rng.beta(a=0.5, b=0.5, size=config.n_snv_loci)
        
        # 2. Follow-up t1 SNVs (Derivato da t0)
        if responder == "Responder":
            # Persistenza del ceppo (piccolo drift genomico < 0.18)
            t1_snv = np.clip(t0_snv + rng.normal(0, 0.03, size=config.n_snv_loci), 0, 1)
        else:
            # Sostituzione di ceppo / divergenza elevata (> 0.18)
            shift = rng.uniform(0.2, 0.45, size=config.n_snv_loci) * rng.choice([-1, 1], size=config.n_snv_loci)
            t1_snv = np.clip(t0_snv + shift, 0, 1)

        t0_id, t1_id = f"{pid}_t0", f"{pid}_t1"
        
        meta_records.append({"sample_id": t0_id, "patient_id": pid, "timepoint": "t0", "clinical_response": responder})
        meta_records.append({"sample_id": t1_id, "patient_id": pid, "timepoint": "t1", "clinical_response": responder})

        snv_dict[t0_id] = t0_snv
        snv_dict[t1_id] = t1_snv

        # 3. Virome Abundances (con segnale di co-occorrenza iniettato)
        t0_vc = rng.lognormal(mean=-2.0, sigma=0.8, size=config.n_viral_clusters)
        t1_vc = rng.lognormal(mean=-2.0, sigma=0.8, size=config.n_viral_clusters)

        # Accoppiamento diretto tra i primi 5 SNVs e i primi 5 Cluster Virali
        t0_vc[:5] += t0_snv[:5] * 2.5
        t1_vc[:5] += t1_snv[:5] * 2.5

        if responder == "NonResponder":
            t1_vc[:5] *= 2.0  # Espansione dei fagi litici nei Non-Responders

        virome_dict[t0_id] = t0_vc / t0_vc.sum()
        virome_dict[t1_id] = t1_vc / t1_vc.sum()

    df_meta = pd.DataFrame(meta_records).set_index("sample_id")
    df_snv = pd.DataFrame.from_dict(snv_dict, orient="index", columns=snv_cols)
    df_virome = pd.DataFrame.from_dict(virome_dict, orient="index", columns=vc_cols)

    df_snv.to_csv(raw_dir / "snv_frequencies.csv")
    df_virome.to_csv(raw_dir / "virome_abundances.csv")
    df_meta.to_csv(meta_dir / "longitudinal_metadata.csv")

    print(f"[INFO] Longitudinal cohort generated ({len(df_meta)} samples across t0 and t1).")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    generate_longitudinal_dataset(
        SimulationConfig(),
        raw_dir=project_root / "data" / "raw",
        meta_dir=project_root / "data" / "metadata"
    )


if __name__ == "__main__":
    main()
