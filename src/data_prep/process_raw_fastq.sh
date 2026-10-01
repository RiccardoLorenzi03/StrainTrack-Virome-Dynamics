#!/usr/bin/env bash
# Upstream Bioinformatic Pipeline: Raw FASTQ -> Viral Abundance Matrix
set -e

FASTQ_DIR="${1:-data/raw_fastq}"
OUTPUT_DIR="${2:-data/raw}"
THREADS="${3:-8}"

mkdir -p "$OUTPUT_DIR" tmp_virome/qc tmp_virome/assembly tmp_virome/genomad tmp_virome/checkv tmp_virome/mapping

echo "=== STEP 1: Quality Control & Filtering (fastp) ==="
for R1 in "$FASTQ_DIR"/*_R1.fastq.gz; do
  SAMPLE=$(basename "$R1" _R1.fastq.gz)
  R2="$FASTQ_DIR/${SAMPLE}_R2.fastq.gz"
  
  fastp -i "$R1" -I "$R2" \
        -o "tmp_virome/qc/${SAMPLE}_clean_R1.fq.gz" \
        -O "tmp_virome/qc/${SAMPLE}_clean_R2.fq.gz" \
        --thread "$THREADS" -q 20 -u 30
done

echo "=== STEP 2: Assembly (MEGAHIT) ==="
for R1 in tmp_virome/qc/*_clean_R1.fq.gz; do
  SAMPLE=$(basename "$R1" _clean_R1.fq.gz)
  R2="tmp_virome/qc/${SAMPLE}_clean_R2.fq.gz"
  
  megahit -1 "$R1" -2 "$R2" \
          -o "tmp_virome/assembly/${SAMPLE}_megahit" \
          -t "$THREADS" --min-contig-len 1500
done

cat tmp_virome/assembly/*_megahit/final.contigs.fa > tmp_virome/all_contigs.fasta

echo "=== STEP 3: Viral Identification (geNomad & CheckV) ==="
genomad end-to-end tmp_virome/all_contigs.fasta tmp_virome/genomad genomad_db/ --threads "$THREADS"

checkv end-to-end tmp_virome/genomad/all_contigs_summary/all_contigs_virus.fna \
                 tmp_virome/checkv -t "$THREADS"

awk -F'\t' '$10=="Complete" || $10=="High-quality" || $10=="Medium-quality" {print $1}' \
    tmp_virome/checkv/quality_summary.tsv > tmp_virome/hq_viral_ids.txt

seqkit grep -f tmp_virome/hq_viral_ids.txt \
            tmp_virome/genomad/all_contigs_summary/all_contigs_virus.fna \
            > tmp_virome/vOTU_references.fasta

echo "=== STEP 4: Read Mapping & Quantification ==="
bowtie2-build tmp_virome/vOTU_references.fasta tmp_virome/mapping/votu_index

for R1 in tmp_virome/qc/*_clean_R1.fq.gz; do
  SAMPLE=$(basename "$R1" _clean_R1.fq.gz)
  R2="tmp_virome/qc/${SAMPLE}_clean_R2.fq.gz"
  
  bowtie2 -x tmp_virome/mapping/votu_index \
          -1 "$R1" -2 "$R2" \
          -p "$THREADS" | samtools view -bS - | samtools sort -o "tmp_virome/mapping/${SAMPLE}.bam"
done

coverm contig --bams tmp_virome/mapping/*.bam \
              --methods relative_abundance \
              -o "$OUTPUT_DIR/virome_abundances.csv"

rm -rf tmp_virome
echo "[SUCCESS] Raw FASTQ processing complete. Matrix written to $OUTPUT_DIR/virome_abundances.csv"
