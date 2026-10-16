#!/usr/bin/env bash
#SBATCH --job-name=germeval-gemma-qlora
#SBATCH --output=results/runs/slurm_%j.out
#SBATCH --error=results/runs/slurm_%j.err
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

echo "=== GermEval 2017 Gemma-2-2B-IT Prompting vs. LoRA vs. QLoRA Suite ==="
echo "Host: $(hostname) | Date: $(date -u)"

# 0. Ensure environment and canonical/derived datasets exist
uv sync --extra dev --extra ml --extra cuda
uv run python -m germeval_llm.pipeline

# 1. Prompting Baselines on Dev & Test Splits (P0, P1, P2 2/5/10-shot, P3)
for SPLIT in dev test_syn test_dia; do
    uv run python -m germeval_llm.cli infer --run-id P0_minimal_0shot --task C --split "$SPLIT" --prompt-key p0_minimal --n-shots 0 --load-in-4bit
    uv run python -m germeval_llm.cli infer --run-id P1_guidelines_0shot --task C --split "$SPLIT" --prompt-key p1_guidelines --n-shots 0 --load-in-4bit
    uv run python -m germeval_llm.cli infer --run-id P3_dev_optimized_10shot --task C --split "$SPLIT" --prompt-key p1_guidelines --n-shots 10 --shot-strategy stratified_diverse --load-in-4bit
done

# 2. Train 4-bit NF4 QLoRA (Q1) and 16-bit bf16 LoRA (L1) on Task C
uv run python -m germeval_llm.cli train --config configs/qlora.yaml --task C --output-dir models/adapters/task_c_qlora_r16_seed42
uv run python -m germeval_llm.cli train --config configs/lora.yaml --task C --output-dir models/adapters/task_c_lora_r16_seed42

# 3. Evaluate Trained Adapters on Dev, Synchronic Test, and Diachronic Test
for SPLIT in dev test_syn test_dia; do
    uv run python -m germeval_llm.cli infer --run-id Q1_qlora_r16 --task C --split "$SPLIT" --adapter-path models/adapters/task_c_qlora_r16_seed42 --load-in-4bit
    uv run python -m germeval_llm.cli infer --run-id L1_lora_r16 --task C --split "$SPLIT" --adapter-path models/adapters/task_c_lora_r16_seed42
done

# 4. Task D Target-String Extraction (D1 Prompting vs. D2 QLoRA)
uv run python -m germeval_llm.cli train --config configs/qlora.yaml --task D --output-dir models/adapters/task_d_qlora_r16_seed42
for SPLIT in dev test_syn; do
    uv run python -m germeval_llm.cli infer --run-id D1_prompt_5shot --task D --split "$SPLIT" --prompt-key d1_guidelines --n-shots 5 --load-in-4bit
    uv run python -m germeval_llm.cli infer --run-id D2_qlora_r16 --task D --split "$SPLIT" --prompt-key d1_guidelines --adapter-path models/adapters/task_d_qlora_r16_seed42 --load-in-4bit
done

echo "=== All GPU Experiments Complete ==="
