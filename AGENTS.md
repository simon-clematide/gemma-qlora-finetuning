# Working in this repository

## Purpose and structure

This Python 3.11+ project compares prompting, LoRA, and QLoRA with Gemma Instruct
on GermEval 2017 aspect-based sentiment analysis. It also provides teaching
notebooks and recorded classroom experiments. The default checkpoint is
`google/gemma-2-2b-it`.

Start with `README.md` for usage and `finetuning-demo.md` for the experimental
specification and dataset audit. Verify behavior against source and CLI help:
some examples and planned features in those documents differ from the code.

The package is under `src/germeval_llm/`:

| Area | Files |
| --- | --- |
| Records and label inventory | `schemas.py`, `taxonomy.py` |
| XML preservation and audits | `xml_parser.py`, `validate.py` |
| Instruction datasets and prompts | `convert.py`, `prompts.py` |
| Output validation and target alignment | `parsing.py`, `alignment.py` |
| Models and adapter training | `inference.py`, `training.py` |
| Scoring and official evaluator integration | `evaluation.py`, `official_evaluation.py` |
| Figures, orchestration, and teaching artifacts | `reporting.py`, `pipeline.py`, `build_notebooks.py`, `build_presentation.py` |

Configuration lives in `configs/`. Check the calling code before assuming a
setting is consumed: not every configuration field drives the current CLI.

## Development commands

Use `uv` with the repository's `.venv` and `uv.lock`. Run commands from the
repository root because dataset, configuration, and output paths are relative.
Use ordinary `make` commands.

```sh
uv sync --extra dev
uv sync --extra dev --extra ml       # When working on model execution
make test                           # Full pytest suite
uv run pytest tests/test_alignment.py -v
uv run pytest --ignore=tests/test_official_evaluation.py -v
uv run python -m germeval_llm.cli --help
```

The CLI provides `infer`, `train`, and `evaluate`. Few-shot inference uses
`--n-shots`. The default prompt follows `--task`: `p1_guidelines` for Task C and
`d1_guidelines` for Task D. Few-shot demonstrations use the matching task’s
training examples.

Keep changes within the relevant modules and follow their existing typed Python
and Pydantic v2 conventions. Add regression coverage for changed parsing,
alignment, scoring, or training behavior. Ruff is a development dependency, but
there is no configured lint or type-check target; avoid unrelated formatting.

## Preparing data and managing outputs

A fresh clone includes `data/canonical/*.jsonl.gz`. The canonical loader can
expand these archives without the historical raw XML checkout. To materialize
missing Task C and Task D datasets:

```sh
uv run python -c 'from germeval_llm.convert import ensure_derived_datasets_ready; ensure_derived_datasets_ready()'
```

This helper fills missing files only. Changes to templates or conversion logic
require deliberate regeneration of the affected derived files.

For workflows requiring original XML, TSV, or the Java evaluator, arrange the
historical checkout and create links with an absolute source path:

```sh
make setup-data HISTORICAL_REPO=/absolute/path/to/historical/repo
```

Check that the resulting symlinks resolve. `make data` regenerates canonical data,
audits, derived datasets, and dataset figures from raw XML. It does not run models
or benchmark simulations. The GPU script instead uses
`ensure_derived_datasets_ready()` to prepare missing datasets from the bundled
canonical archives.

Treat outputs from `run_m3_to_m6_benchmark_artifacts()` as simulations. They are
written exclusively under `results/simulations/`; the pipeline CLI requires
`--simulate` to generate them. Never present simulated scores as model inference
measurements. Existing artifacts in legacy `results/` paths can still be synthetic;
keep them distinct from recorded classroom replay and actual model executions.

Respect `.gitignore`: raw inputs, expanded datasets, and adapter weights are
local artifacts. Compressed canonical archives and selected results/notebooks
are intentionally tracked. Regenerate only artifacts needed for the task and
review their diffs. Keep credentials out of source, notebooks, logs, and commits.

## Contracts to preserve

- Canonical documents retain exact text, split and source order, all opinions,
  XML attributes, and original offsets. Preserve duplicates and overlapping
  annotations; maintain lossless XML round-trip checks.
- Correct annotation typos only under an explicit derivation or scoring policy.
  Never modify canonical gold to normalize it. Dataset configuration enables
  training-split patches by default and disables them for other splits.
- Include irrelevant and empty-opinion documents in training and evaluation.
  Their empty gold sets are meaningful; strict scoring penalizes false positives.
- Task C produces `{"labels": ["Zugfahrt:negative"]}` or `{"labels": []}`.
  Use taxonomy helpers for validation, ordering, and deduplication. Preserve
  different polarities attached to the same category. C1 measures categories;
  C2 measures category/polarity pairs.
- Task D produces `{"targets": [{"target": "Zug", "category": "Zugfahrt", "polarity": "negative"}]}`.
  Model-facing data contains target strings, never character offsets, token
  indices, or `__D__` markers. Exclude implicit `NULL` opinions from extraction
  and retain discontinuous strings such as `"zeigt * an"`.
- `alignment.py` converts Task D strings to zero-based half-open character spans.
  Keep repeated-string disambiguation deterministic, preserve discontinuous
  bounding spans, and count unaligned predictions in error accounting.
- Store raw completions, parse statuses, and validation errors. Preserve the
  distinction between exact JSON compliance and tolerant extraction; do not
  silently repair unknown labels.
- Gemma formatting must use a single BOS and the correct turn markers. Training
  uses completion-only loss: prompt tokens through `<start_of_turn>model\n`
  are masked with `-100`. Truncation must not remove all completion tokens.
- Draw training data and demonstrations from train, tune choices on dev, and
  evaluate frozen configurations on `test_syn` and `test_dia`. Record seeds,
  model/prompt settings, training subsets, evaluation IDs, and actual hardware.

## Evaluation and validation

Maintain separate strict Python and official Java-compatible metrics. The
historical evaluator skips some empty-gold documents and uses nonstandard
F-score aggregation; compatibility code deliberately reproduces these quirks.

Java invocation uses `<task> <PREDICTIONS> <GOLD>`. Task D `OTE` ignores polarity
and secondary offsets and includes implicit opinions when present in raw gold.
Label explicit-only and raw-gold evaluations clearly.

Use `align_predictions()` and `verify_experiment_manifest()` for model
comparisons and replay. Check document IDs, splits, and manifest settings before
comparing scores; missing or mismatched predictions must not silently disappear.

Run the relevant existing tests after code changes. Full-suite Java integration
requires `data/raw/EvaluationScript.jar` and a Java runtime; `OTE` needs JAXB,
normally supplied by Java 8. Those tests do not automatically skip missing
prerequisites. Report unavailable checks explicitly. Python scoring and ordinary
unit tests need neither Java nor downloaded model weights.

Actual model runs require the ML dependencies, gated Gemma access, and suitable
hardware. NF4 quantization requires CUDA and bitsandbytes; non-CUDA paths load
unquantized weights. Bound smoke runs with `--max-docs`, `--subset-size`, and
`--max-steps` as appropriate. Do not treat CPU/MPS validation as proof that the
CUDA quantization path works.

## Notebooks and teaching materials

Edit notebook `00`–`10` cell specifications in `build_notebooks.py`, their source
of truth. `make notebooks` rewrites them without executing cells and replaces
existing outputs. Execute deliberately with:

```sh
uv run python -m germeval_llm.build_notebooks --execute
```

Prepare the required data/results first and review generated notebook diffs.
Notebook `11_colab_l4_live_demo.ipynb` is maintained separately. Preserve its
live/replay distinction and matching result manifests. Its regression test
checks recorded outputs, figures, prediction alignment, and comparison results;
clearing outputs can invalidate that test.

Notebook 11's default replay bundle is
`results/replays/2026-10-18_mps_lora_n1000_seed42/manifest.json`. Use
`load_replay_manifest()` to verify its hashes and recorded settings. It is a
measured MPS LoRA recording, independent of the live CUDA quantization switch.
The compact bundle is intended for Git; full weights, tokenized datasets, and
source snapshots remain local under ignored `results/recorded_runs/`. Refresh
notebook 11's outputs by deliberately executing replay, then verify its regression
test and the rendered figures.
