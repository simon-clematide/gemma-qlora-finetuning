# GermEval 2017 ABSA: Prompting vs. LoRA vs. QLoRA (Gemma Instruct)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simon-clematide/gemma-qlora-finetuning/blob/main/notebooks/11_colab_l4_live_demo.ipynb)

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
│   └── cli.py                      # Command-line interface (infer, train, evaluate)
├── notebooks/                      # 11 self-contained, executed CAS teaching notebooks (00-10)
├── scripts/
│   └── run_gpu_experiments.sh      # Automated batch runner for full GPU experiments
└── tests/                          # Unit and official Java evaluator integration tests
```

---

## Setup & Installation

This project uses [`uv`](https://docs.astral.sh/uv/) with a local `.venv/` inside the project directory.

```bash
# 1. Create/sync the local .venv with core, dev, and ML (PyTorch/Transformers/PEFT/TRL) packages
uv sync --extra dev --extra ml

# 2. Prepare Task C/D datasets from the bundled canonical archives (no raw XML or Java needed)
uv run python -c 'from germeval_llm.convert import ensure_derived_datasets_ready; ensure_derived_datasets_ready()'

# 3. Optional: link historical XML/TSV files and the evaluator, then regenerate data and audits
make setup-data HISTORICAL_REPO=/absolute/path/to/historical/repo
make data HISTORICAL_REPO=/absolute/path/to/historical/repo

# 4. Run the full unit and official Java evaluator reconciliation test suite
uv run pytest -v
```

> **Note on Hugging Face Gated Model Access:**
> `google/gemma-2-2b-it` is a gated model on the Hugging Face Hub. Ensure you have accepted the license on Hugging Face and authenticated locally (`uv run hf auth login` or `HF_TOKEN` environment variable).
>
> **Note on Java Evaluator (`EvaluationScript.jar`):**
> `EvaluationScript.jar OTE` (Task D XML evaluation) uses `javax.xml.bind` (JAXB), which requires Java 8. `src/germeval_llm/official_evaluation.py` automatically detects a local Java 8 installation (e.g., `/Library/Internet Plug-Ins/JavaAppletPlugin.plugin/Contents/Home/bin/java` on macOS) when running `OTE` mode.

---

## Data preparation and simulated teaching artifacts

`make data` regenerates canonical data, audits, derived datasets, and dataset figures from the historical XML. It does not run models or generate benchmark simulations.

To explicitly generate synthetic teaching benchmarks as well, run:

```bash
uv run python -m germeval_llm.pipeline --simulate
```

Simulation predictions, metrics, run manifests, and figures are isolated under `results/simulations/`, with simulation provenance recorded there. They require the historical inputs and Java evaluator. They are not measurements from Gemma inference. Existing benchmark tables and figures in the legacy `results/` paths may contain synthetic artifacts; do not interpret them as actual model results. The historical comparison numbers below are illustrative, not validated measurements for the current configuration. Notebook 11 replays the measured 18 October 2026 MPS LoRA run from `results/replays/2026-10-18_mps_lora_n1000_seed42/`, including verified predictions, training history, and provenance.

Actual CLI inference writes per-split predictions and metrics under `results/predictions/` and `results/metrics/`, with matching model, prompt, hardware, and evaluation manifests under `results/runs/`. It does not rebuild aggregate benchmark tables. Only compare matching prediction sets and experiment manifests.

## Fresh classroom run with full provenance

With the ML dependencies, a CUDA/MPS accelerator, and `google/gemma-2-2b-it` already cached, run:

```bash
PYTHONHASHSEED=42 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python scripts/run_recorded_classroom.py \
  --run-dir results/recorded_runs/my_fresh_classroom_run
```

This generates both prediction sets from scratch, trains a new LoRA adapter for 100 steps on 1,000 examples, and evaluates the first 150 `test_syn` documents. The 1,152-token training limit preserves the complete selected targets; preparation fails if examples or completions would be truncated. It records the pinned model revision, source snapshot, exact train/dev/test IDs, token-level loss masks, raw prompts and responses, training history, actual settings and dtypes, saved adapter, reload checks, independent scoring, figures, and SHA-256 inventory. Complete bundles are local artifacts ignored by Git. This is a single-seed classroom experiment, not a CUDA QLoRA benchmark.

## Running Inference & Fine-Tuning via CLI

### 1. Prompting Inference (`P0`, `P1`, `P2`, `P3`, `D1`)

Zero- and few-shot CLI inference for both tasks batches similar-length prompts, checks that prompts plus the output budget fit the model context, and restores source order in predictions. CUDA OOM recovery clears unused cache and retries once before reducing the batch size. It never silently drops documents, shortens text, or changes the demonstrations.

Completed batches are atomically saved under ignored `results/recorded_runs/inference/`. Rerun the same command to resume; model/adapter, precision, prompt/demo, dataset, or batching changes are rejected rather than mixed with earlier results. Choose a new `--run-id` or `--checkpoint-path` for a different experiment. `--batch-size` overrides automatic GPU defaults; `--no-length-grouping` disables grouping, and `--no-checkpoint` disables checkpointing. Use `--model-revision` to pin a model. Checkpoints survive a kernel restart only while the files remain; on Colab, copy the folder to Drive to survive deletion of the virtual machine.


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
  --prompt-key p1_guidelines \
  --n-shots 5
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

On a CUDA GPU workstation or server (e.g., NVIDIA L4, RTX 3090/4090, or A100), run the implemented experiments: `P0`, `P1`, fixed 10-shot `P3`, `L1`, and `Q1` on dev and both test splits, plus `D1–D2` on dev and `test_syn`. The script does not run P2, learning curves, rank ablations, or joint C+D training:

```bash
bash scripts/run_gpu_experiments.sh
```

---

## CAS Teaching Notebooks (`notebooks/`)

Notebooks 00–10 are generated without execution by `make notebooks`. This replaces their existing outputs. To execute them deliberately after preparing the required data and results, use `uv run python -m germeval_llm.build_notebooks --execute`. Newly generated notebooks 04–10 read isolated simulation artifacts and label them as synthetic teaching examples. Existing notebook files retain their recorded outputs until deliberately regenerated. Notebook 11 is maintained separately; the generator leaves it untouched.

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

## Replaying the recorded classroom experiment

Notebook 11 uses the fresh **MPS LoRA** recording: 1,000 training examples, 100 optimizer steps, 150 fixed test documents, and about 92 minutes of recorded training. Its compact replay bundle is included under `results/replays/2026-10-18_mps_lora_n1000_seed42/`; the complete local weights archive is not required. Replay verifies artifact hashes and recorded settings, then recreates the tables and figures without model downloads or training.

Set `REQUESTED_MODE = 'replay'` in the first code cell to replay on any machine, including a CUDA runtime. `auto` runs live on CUDA and replays otherwise. The live quantization switch does not change the recorded LoRA method or replay labels. Live runs generate a new baseline using their selected precision and the pinned model revision.

## Running Live on Google Colab Pro (NVIDIA L4 GPU)

The repository is pre-configured for **one-click execution on Google Colab** without uploading raw XMLs or running XML preprocessing live:
- All 26,209 canonical GermEval 2017 documents are tracked in Git as compact gzipped archives (`data/canonical/*.jsonl.gz`, `~5 MB` total).
- On first import, [`ensure_derived_datasets_ready()`](src/germeval_llm/convert.py) automatically decompresses the canonical splits and materializes `data/derived/task_c/*.jsonl` and `data/derived/task_d/*.jsonl` in ~1.5 seconds.
- All Task C (`C1`, `C2`, strict + official Java-compatible) and Task D (`string_multiset`, `span_explicit_exact`, `span_explicit_overlap`) metrics run in **100% pure Python** ([`src/germeval_llm/evaluation.py`](src/germeval_llm/evaluation.py)) with no Java runtime required on Colab.
- Open [`notebooks/11_colab_l4_live_demo.ipynb`](notebooks/11_colab_l4_live_demo.ipynb) in Google Colab, select an **L4 GPU** runtime, and click **Run All**.

---

## 16-bit `bfloat16` LoRA vs. 4-bit `nf4` QLoRA: When to Use Which & How to Configure

You can switch seamlessly between **16-bit LoRA** (`load_in_4bit: false`) and **4-bit NormalFloat QLoRA** (`load_in_4bit: true`) in both the CLI and the Colab notebook:

### 1. Memory & Speed Trade-Off on `google/gemma-2-2b-it` (2.61B Parameters)

| GPU VRAM Component | Full Fine-Tuning (`bf16`) | **16-bit LoRA (`configs/lora.yaml`)** | **4-bit NF4 QLoRA (`configs/qlora.yaml`)** |
|---|---|---|---|
| **Base Model Weights ($W_0$)** | `~5.0 GB` (16 bits/param) | **`~5.0 GB`** (16 bits/param, frozen) | **`~1.4 GB`** (~4.5 bits/param, frozen) |
| **Trainable Parameters** | `2,614M` (`100%`) | **`20.8M`** (`0.79%`, LoRA $A,B$ in `bf16`) | **`20.8M`** (`0.79%`, LoRA $A,B$ in `bf16`) |
| **Gradients + Optimizer States** | `~26.0 GB` | **`~0.1 GB`** | **`~0.1 GB`** |
| **Activations (with grad ckpt)** | `~2.5 GB` | **`~2.1 GB`** | **`~1.4 GB`** |
| **Peak Training GPU VRAM** | **`> 33 GB`** (OOM on L4 24GB) | **`~7.2 GB`** | **`~2.9 GB`** |
| **Relative Step Speed** | Baseline | **Faster (~20% faster than 4-bit)** | Slightly slower (on-the-fly NF4 dequantization) |
| **Illustrative Task C2 Strict Micro-F1 (`test_syn`)** | — | **`0.8384`** | **`0.8360`** (`< 0.3` F1 pt difference) |

### 2. Practical Guidance
- **When to use 16-bit `bfloat16` LoRA (`USE_4BIT_QUANTIZATION = False` / `configs/lora.yaml`):**
  - On an **NVIDIA L4 (24 GB VRAM)** or **Apple Silicon MPS**, `google/gemma-2-2b-it` in native 16-bit `bfloat16` requires only `~7.2 GB` of training VRAM.
  - Because it does not need `bitsandbytes` or on-the-fly 4-bit dequantization during every forward/backward matrix multiplication, **16-bit LoRA is simpler and ~15–25% faster per step**. The notebook’s replay uses this recorded LoRA method; live execution has a separate quantization switch.
- **When to use 4-bit `nf4` QLoRA (`USE_4BIT_QUANTIZATION = True` / `configs/qlora.yaml`):**
  - When GPU VRAM is constrained (e.g., free 15 GB Colab T4 with long sequences/large batches) or when scaling up to **9B–27B models** (`google/gemma-2-9b-it` or `google/gemma-3-12b-it`), where 16-bit base weights alone consume `18–24+ GB` VRAM.
  - NormalFloat4 (`nf4`) places its 16 quantization bins at equal-probability quantiles of $\mathcal{N}(0, \sigma^2)$ while keeping LoRA adapters ($A, B$) and matrix multiplications in `bfloat16`, cutting base weight memory by **~72%** with virtually zero accuracy loss.

### 3. How to Switch Between 16-bit and 4-bit
- **In [`notebooks/11_colab_l4_live_demo.ipynb`](notebooks/11_colab_l4_live_demo.ipynb) (first code cell):**
  ```python
  USE_4BIT_QUANTIZATION = False  # False -> 16-bit bfloat16 LoRA (configs/lora.yaml)
                                 # True  -> 4-bit NF4 QLoRA     (configs/qlora.yaml)
  ```
- **In Python (`load_gemma_model_and_tokenizer` / `run_sft_adapter_training`):**
  - Pass `load_in_4bit=False` (16-bit) or `load_in_4bit=True` (4-bit) to `load_gemma_model_and_tokenizer(...)`.
  - Pass `config_path="configs/lora.yaml"` (16-bit) or `config_path="configs/qlora.yaml"` (4-bit) to `run_sft_adapter_training(...)`.
- **In the CLI:**
  - Omit `--load-in-4bit` on `infer` and pass `--config configs/lora.yaml` on `train` for 16-bit mode.


