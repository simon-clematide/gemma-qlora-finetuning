# GermEval 2017 ABSA: Prompting vs. LoRA vs. QLoRA (Gemma Instruct)

Reproducible CAS fine-tuning demonstration comparing zero-shot prompting (`P0`, `P1`), few-shot in-context learning (`P2`, `P3`), 16-bit LoRA (`L1`), and 4-bit NF4 QLoRA (`Q1–Q3`, `D2`, `J1`) on **GermEval 2017 Shared Task on Aspect-based Sentiment in Social Media Customer Feedback**:
- **Task C (Core):** Document-level multi-label aspect category (`C1`, 19 categories) and aspect+polarity (`C2`, 57 `Category:Polarity` pairs) classification.
- **Task D (Extension):** Opinion Target Expression (OTE) extraction via **verbatim target-string generation** followed by deterministic software span alignment (`alignment.py`) for official XML evaluation.

See [`finetuning-demo.md`](finetuning-demo.md) for the complete implementation specification, empirical dataset audit, Java evaluator analysis, and experimental design.

---

## Core Design Principles

1. **Strict Offset-Free LLM Boundary:**
   The LLM is **never** prompted or trained to output character offsets (`from`/`to`) or token indices (`__D__`).
   - **Task C output contract:** `{"labels": ["Allgemein:neutral", "Zugfahrt:negative"]}` (or `{"labels": []}` when `relevance=false` / no opinion is expressed).
   - **Task D output contract:** `{"targets": [{"target": "Regionalbahn", "category": "Zugfahrt", "polarity": "negative"}]}` (preserving `" * "` for discontinuous German multi-word targets such as `"zeigt * an"`).
   - Deterministic software in [`src/germeval_llm/alignment.py`](src/germeval_llm/alignment.py) maps predicted target strings back to 0-based `[from, to)` character spans for official `EvaluationScript.jar OTE` scoring.
2. **Dual Strict & Legacy-Compatible Evaluation:**
   - **Strict Python Evaluation (`src/germeval_llm/evaluation.py`):** Evaluates all documents (including the ~16–18% `relevance=false` documents with `{"labels": []}`) using standard Micro-F1, Macro-F1, Exact-Set Accuracy, and paired bootstrap 95% confidence intervals ($B=1000$).
   - **Official Java Reconciliation (`src/germeval_llm/official_evaluation.py`):** Exports predictions to official GermEval 2017 TSV (`category`) and XML (`OTE`) formats, runs `EvaluationScript.jar`, and reconciles exact numerical agreement (accounting for the historical Java evaluator's empty-gold-document loop and `Fscore.java`'s dummy `"none"` label Jaccard calculation).

---

## Repository Structure

```text
.
├── finetuning-demo.md              # Full technical specification & empirical audit
├── pyproject.toml                  # Package metadata & dependencies (uv)
├── Makefile                        # Reproducible build, test, and notebook targets
├── configs/
│   ├── dataset.yaml                # Raw XML/TSV paths, SHA256, typo-patching rules
│   ├── model.yaml                  # google/gemma-2-2b-it model & decoding config
│   ├── prompts.yaml                # P0, P1, P2, P3, Task D, and Joint C+D templates
│   ├── lora.yaml                   # 16-bit LoRA hyperparameters (r=16, alpha=32)
│   ├── qlora.yaml                  # 4-bit NF4 QLoRA hyperparameters (double quant, bf16)
│   └── experiments.yaml            # Full experiment matrix (P0-P3, L1, Q1-Q3, D1-D2, J1)
├── src/germeval_llm/
│   ├── schemas.py                  # Pydantic v2 models for canonical, Task C, Task D, and run records
│   ├── taxonomy.py                 # 19-category GermEval taxonomy & German guideline descriptions
│   ├── xml_parser.py               # Lossless XML -> canonical JSONL parser & XML round-tripper
│   ├── validate.py                 # Dataset anomaly & typo audit (positve, Sichaupt, QR-Code, etc.)
│   ├── prompts.py                  # Gemma chat template formatter & completion-only loss mask verifier
│   ├── convert.py                  # Canonical -> Task C / Task D instruction JSONL + stratified sampler
│   ├── parsing.py                  # Strict & tolerant JSON parser with schema/taxonomy validation
│   ├── alignment.py                # Deterministic Task D target-string -> character-span aligner
│   ├── evaluation.py               # Strict Python C1/C2/D scorer, Java-compat scorer, bootstrap CIs
│   ├── official_evaluation.py      # TSV/XML exporter & subprocess bridge to EvaluationScript.jar
│   ├── inference.py                # Gemma Instruct model loader & greedy generation runner
│   ├── training.py                 # PEFT LoRA/QLoRA + TRL SFTTrainer with completion_only_loss=True
│   ├── reporting.py                # Publication-quality matplotlib figures & summary tables
│   ├── pipeline.py                 # End-to-end deterministic data & benchmark artifact pipeline
│   ├── build_notebooks.py          # Generator & executor for notebooks 00 through 10
│   └── cli.py                      # Command-line interface (prepare-data, infer, train, build-notebooks)
├── notebooks/                      # 11 self-contained, executed CAS teaching notebooks (00-10)
├── scripts/
│   └── run_gpu_experiments.sh      # Automated batch runner for full GPU experiments
└── tests/                          # 14 unit and official Java evaluator integration tests
```

---

## Setup & Installation

This project uses [`uv`](https://docs.astral.sh/uv/) with a local `.venv/` inside the project directory.

```bash
# 1. Create/sync the local .venv with core, dev, and ML (PyTorch/Transformers/PEFT/TRL) packages
uv sync --extra dev --extra ml

# 2. Link historical GermEval 2017 XML/TSV files and EvaluationScript.jar into data/raw/
make setup-data

# 3. Run canonical XML conversion, validation audit, and Task C/D instruction dataset derivation
make data

# 4. Run the full unit and official Java evaluator reconciliation test suite
uv run pytest -v
```

> **Note on Hugging Face Gated Model Access:**
> `google/gemma-2-2b-it` is a gated model on the Hugging Face Hub. Ensure you have accepted the license on Hugging Face and authenticated locally (`uv run hf auth login` or `HF_TOKEN` environment variable).
>
> **Note on Java Evaluator (`EvaluationScript.jar`):**
> `EvaluationScript.jar OTE` (Task D XML evaluation) uses `javax.xml.bind` (JAXB), which requires Java 8. `src/germeval_llm/official_evaluation.py` automatically detects a local Java 8 installation (e.g., `/Library/Internet Plug-Ins/JavaAppletPlugin.plugin/Contents/Home/bin/java` on macOS) when running `OTE` mode.

---

## Running Inference & Fine-Tuning via CLI

### 1. Prompting Inference (`P0`, `P1`, `P2`, `P3`, `D1`)

Run greedy inference with `google/gemma-2-2b-it` on any split (`dev`, `test_syn`, `test_dia`). On Apple Silicon (`mps`) or CPU, pass `--max-docs` for a fast smoke test:

```bash
# Fast 5-document zero-shot smoke test on dev (runs on CUDA or Apple Silicon MPS)
uv run python -m germeval_llm.cli infer \
  --run-id smoke_p1_dev \
  --task C \
  --split dev \
  --prompt-key p1_guidelines \
  --max-docs 5

# 5-shot in-context learning (P3) on full dev split
uv run python -m germeval_llm.cli infer \
  --run-id p3_dev_opt \
  --task C \
  --split dev \
  --prompt-key p3_dev_optimized \
  --num-shots 5
```

### 2. LoRA & QLoRA Fine-Tuning (`L1`, `Q1`, `Q2`, `D2`)

Train a PEFT adapter using `trl.SFTTrainer` with `completion_only_loss=True` (so 100% of prompt tokens up to `<start_of_turn>model\n` are masked with `-100`):

```bash
# Fast 5-step LoRA smoke training run (verified on Apple Silicon MPS & CUDA)
uv run python -m germeval_llm.cli train \
  --config configs/lora.yaml \
  --task C \
  --output-dir models/adapters/smoke_task_c_lora \
  --subset-size 64 \
  --max-steps 5

# Reload the trained adapter and run inference
uv run python -m germeval_llm.cli infer \
  --run-id smoke_lora_dev \
  --task C \
  --split dev \
  --prompt-key p1_guidelines \
  --adapter-path models/adapters/smoke_task_c_lora \
  --max-docs 5

# Full 4-bit NF4 QLoRA training on Task C (requires CUDA GPU with bitsandbytes)
uv run python -m germeval_llm.cli train \
  --config configs/qlora.yaml \
  --task C \
  --output-dir models/adapters/q1_qlora_r16
```

### 3. Full GPU Experiment Suite (`scripts/run_gpu_experiments.sh`)

On a CUDA GPU workstation or server (e.g., NVIDIA L4, RTX 3090/4090, or A100), run the entire experimental matrix (`P0–P3`, `L1`, `Q1`, the 18-run `Q2` learning-curve grid across 6 subset sizes $\times$ 3 seeds, and `D1–D2` Task D extraction + alignment):

```bash
bash scripts/run_gpu_experiments.sh
```

---

## CAS Teaching Notebooks (`notebooks/`)

All 11 notebooks are generated and executed via `make notebooks` (`uv run python -m germeval_llm.cli build-notebooks`):

| Notebook | Topic & Pedagogical Goal |
|---|---|
| [`00_environment.ipynb`](notebooks/00_environment.ipynb) | Hardware/GPU/MPS & Java 8 verification, dataset SHA256 manifest check |
| [`01_dataset_exploration.ipynb`](notebooks/01_dataset_exploration.ipynb) | GermEval 2017 split statistics, 19 categories, implicit vs. explicit targets, discontinuous `*` spans |
| [`02_xml_conversion.ipynb`](notebooks/02_xml_conversion.ipynb) | Lossless XML $\rightarrow$ canonical JSONL parsing, round-trip verification, raw annotation typo audit |
| [`03_instruction_dataset.ipynb`](notebooks/03_instruction_dataset.ipynb) | Task C & Task D JSON contracts, Gemma chat formatting, token lengths, `-100` completion-only loss masking |
| [`04_prompting_baselines.ipynb`](notebooks/04_prompting_baselines.ipynb) | Zero-shot (`P0`, `P1`) vs. few-shot (`P2`, `P3`) accuracy, schema compliance, and token cost |
| [`05_lora_finetuning.ipynb`](notebooks/05_lora_finetuning.ipynb) | 16-bit LoRA architecture ($W + \frac{\alpha}{r}BA$), Gemma linear modules, parameter & VRAM profile |
| [`06_qlora_finetuning.ipynb`](notebooks/06_qlora_finetuning.ipynb) | 4-bit NF4 double quantization, QLoRA vs. LoRA memory/accuracy trade-off, live MPS smoke verification |
| [`07_learning_curves.ipynb`](notebooks/07_learning_curves.ipynb) | Sample efficiency across $N \in \{100 \dots 19,432\}$, stratified vs. uniform sampling, minimum step floor |
| [`08_task_d_extraction.ipynb`](notebooks/08_task_d_extraction.ipynb) | Verbatim target-string generation + deterministic span alignment (`unique`, `ambiguous`, `discontinuous`) |
| [`09_evaluation.ipynb`](notebooks/09_evaluation.ipynb) | Authoritative benchmark table, strict Python vs. official `EvaluationScript.jar` reconciliation |
| [`10_error_analysis.ipynb`](notebooks/10_error_analysis.ipynb) | Rare-category recall, `relevance=false` false positives, synchronic vs. diachronic generalization |
| [`11_colab_l4_live_demo.ipynb`](notebooks/11_colab_l4_live_demo.ipynb) | **End-to-End Google Colab Pro (L4 GPU) Live Demo:** zero-preprocessing bootstrap, 4-bit NF4 QLoRA training, test-set inference, and pure-Python evaluation |

---

## Running Live on Google Colab Pro (NVIDIA L4 GPU)

The repository is pre-configured for **one-click execution on Google Colab** without uploading raw XMLs or running XML preprocessing live:
- All 26,209 canonical GermEval 2017 documents are tracked in Git as compact gzipped archives (`data/canonical/*.jsonl.gz`, `~5 MB` total).
- On first import, [`ensure_derived_datasets_ready()`](src/germeval_llm/convert.py) automatically decompresses the canonical splits and materializes `data/derived/task_c/*.jsonl` and `data/derived/task_d/*.jsonl` in ~1.5 seconds.
- All Task C (`C1`, `C2`, strict + official Java-compatible) and Task D (`string_multiset`, `span_explicit_exact`, `span_explicit_overlap`) metrics run in **100% pure Python** ([`src/germeval_llm/evaluation.py`](src/germeval_llm/evaluation.py)) with no Java runtime required on Colab.
- Open [`notebooks/11_colab_l4_live_demo.ipynb`](notebooks/11_colab_l4_live_demo.ipynb) in Google Colab, add your `HF_TOKEN` to **Colab Secrets**, select an **L4 GPU** runtime, and click **Run All**.

