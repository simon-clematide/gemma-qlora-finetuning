# Fresh recorded classroom experiment

Run: 2026-10-18_fresh_classroom_lora_seed42

Hardware: mps; LoRA without quantization.
Training: 1,000 train examples, 100 optimizer steps; validation: first 64 dev documents.
Training time: 5,516.52 seconds (about 92 minutes). Adapter dtype: bfloat16.
Sequence limit: 1,152 tokens, preserving every selected target. The initial 1,024-token preparation dropped 10 examples and truncated other completions; that attempt was stopped before a completed optimizer step and archived under `attempts/attempt_01/`. The corrected training started again from the base model with the same seed.
Evaluation: same first 150 test_syn documents; both prediction sets freshly generated.

| Metric | Baseline | Adapter |
| --- | ---: | ---: |
| Usable task output (tolerant parser) | 0.5667 | 0.9800 |
| C2 strict micro-F1 | 0.1008 | 0.5985 |

Paired C2 difference: +0.4978; 95% document-bootstrap interval [+0.4091, +0.5863].

This is one seed and a fixed demonstration slice. The interval does not measure training-seed variability. This run does not measure CUDA QLoRA.

All raw outputs reparse consistently, independent C2 counts match package scores, and both systems received identical formatted prompts. Three selected predictions reproduced exactly after a separate adapter reload. Exact train/dev IDs match the tokenized rows, all saved weights are finite, and the complete bundle has a SHA-256 inventory. See `manifest.json`, `training_summary.json`, `reload_verification.json`, `weights_audit.json`, and `sha256.json`.

## Portable replay package

This directory contains the compact, repository-distributed replay of the source run. It includes the two prediction sets, training history and configuration, comparison, recorded tokenization example, reload and weight audit reports, and provenance. Model weights, full training datasets, and source snapshots remain in the full local `results/recorded_runs/2026-10-18_fresh_classroom_lora_seed42/` bundle and are not needed for replay. `source_manifest.json` and `source_sha256.json` preserve that run’s provenance; `sha256.json` verifies the files actually distributed here.

## Wrapper policy

Consistent Markdown fences are fixable with deterministic postprocessing. The baseline has 144/150 complete fences. Removing complete outer fences yields valid JSON syntax for 143/150 responses and valid task output for 84/150. The existing tolerant parser extracts one additional task object from a malformed array, yielding 85/150 usable outputs (56.7%), compared with 147/150 (98%) for the adapter. C1/C2 already use extracted labels, so fencing itself incurs no label-score penalty. Raw unwrapped-task compliance (0% versus 98%) is retained only as a secondary interface measure, not as evidence that fine-tuning was needed to remove wrappers.
