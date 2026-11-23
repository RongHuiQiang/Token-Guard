#!/bin/bash
# run_paper_impl.sh — efficient paper-faithful implementation of Token-Guard
# (three-stage scoring as described in the paper; decode-time token gating
#  with KV cache; sampling temperature 0.4 per paper §5.2).
#
# Usage:
#   bash run_paper_impl.sh <gpu> <dataset> [n] [start]
#   dataset ∈ {PubmedQA, RAGTruth, CovidQA, Halueval, FinanceBench, DROP_History, DROP_Nfl}
# Example (100 samples of PubmedQA on GPU 0):
#   bash run_paper_impl.sh 0 PubmedQA 100
set -u
GPU="${1:-0}"; DS="${2:-PubmedQA}"; N="${3:-26}"; START="${4:-0}"
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="${TG_MODEL:-$HERE/../models/Meta-Llama-3.1-8B-Instruct}"

declare -A DMAP
DMAP["CovidQA"]="covidQA"; DMAP["DROP_History"]="history"; DMAP["DROP_Nfl"]="nfl"
DMAP["FinanceBench"]="financebench"; DMAP["Halueval"]="halueval"
DMAP["PubmedQA"]="pubmedqa"; DMAP["RAGTruth"]="ragtruth"

OUT="${TG_OUT:-$HERE/../results/paper_impl}/${DS}"; mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=$GPU python tg_paper_impl.py \
  --model_path "$MODEL" \
  --datasets "${DMAP[$DS]}" \
  --data_path "$HERE/../data/${DS}.json" \
  --start_index "$START" --max_examples "$N" \
  --temperature 0.4 --seed 0 \
  --output_dir "$OUT" --file_name "${DS}_s${START}"
