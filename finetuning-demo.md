# GermEval 2017 ABSA: Prompting vs. LoRA vs. QLoRA (Gemma Instruct)

**Detailed implementation specification for a reproducible CAS fine-tuning demonstration**  
**Status:** implementation plan, refined against empirical audit of the historical repository and official evaluator  
**Primary task:** GermEval 2017 Task C, document-level aspect–polarity prediction (both *Category-only* and *Category+Sentiment* sub-metrics)  
**Extension:** Task D, opinion target *text* extraction (OTE)  
**Target model family:** Google Gemma 2 / Gemma 3 Instruct (e.g., `google/gemma-2-2b-it` or `google/gemma-3-4b-it`)  
**Core constraint:** **Never prompt or train the LLM to produce character offsets or token indices.** The model emits copied target strings; deterministic software derives character offsets only when required by the official XML evaluation format.

## 1. Goals and research questions

Build a robust, inspectable demonstration comparing (1) zero-shot prompting, (2) guideline-based prompting, (3) few-shot prompting, (4) LoRA supervised fine-tuning, and (5) 4-bit QLoRA supervised fine-tuning using **the same Gemma Instruct checkpoint**. Evaluate all approaches on identical held-out examples, with the same label inventory, output parser, and scoring code. Do not presume that fine-tuning will outperform strong prompting; the empirical comparison—across accuracy, schema adherence, data efficiency, and token/compute cost—is the lesson.

Questions:

1. How much does supplying the official GermEval category definitions improve a pretrained Gemma Instruct model?
2. How much does in-context demonstration improve performance, and at what inference-token and KV-cache cost?
3. How much does LoRA/QLoRA improve Task C micro-F1 (Category-only and Category+Sentiment), macro-F1, schema adherence, and rare-category recall?
4. What is the relationship between training-set size, performance, memory consumption, and training time—and how do fixed-epoch vs. fixed-step training budgets behave on small subsets?
5. For Task D, can a generative model reliably copy annotated target expressions (including discontinuous German verbal brackets/multi-part targets), and how much performance is lost in deterministic string-to-span alignment?
6. Do improvements transfer from the synchronic (`test_TIMESTAMP1`) to the diachronic (`test_TIMESTAMP2`) evaluation split?

**Out of scope for the core CAS lab:** full-parameter fine-tuning, training a tokenizer, reproducing the historical DyNet architecture, token-index prediction, and prompting for character offsets. The 2018 BiLSTM-CRF/ensembling results serve as an external historical reference rather than a strictly controlled model-size comparison.

## 2. Source material, empirical data audit, and historical compatibility

Use the original GermEval 2017 XML files, not the Penn-formatted training data, as the annotation source of truth. The historical repository is:

- https://github.com/simon-clematide/konvens-2018-german-absa (locally available at `konvens-2018-german-absa`)
- `lib/absaxml2tsv.py`: XML parsing, regex tokenization, offset bookkeeping, multiple labels per token, synthetic `__D__` marker for implicit/document-level annotations.
- `lib/tsv2penn.py`: conversion to single-tag Penn-style sequences; **do not inherit its most-frequent-category reduction**, which discards multilabel information.
- `lib/tsv2evalinput.py` and `lib/tsv2evalinputxml.py`: reference implementations of official-format output conversion; inspect and test, but reimplement cleanly in Python 3 without legacy bugs.
- `Makefile`: original split filenames and evaluation entry points.
- `GermEval2017/EvaluationScript.jar` (and Java source in `GermEval2017/src/`): historical official evaluator, retained as an independent scoring reference.

### 2.1 Verified split availability and empirical audit findings

An upfront inspection of `data/*.xml` and `*.tsv` in the historical repository confirms that **full gold annotations (both XML `<Opinion>` elements and TSV gold files) ARE present for all four splits**:

| Split file | Role | Total docs | Unique IDs | Docs with 0 opinions | Relevance=`false` docs | Total opinions | Implicit (`NULL`, `to=0`) | Explicit opinions |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `train_v1.4.xml` | Training | 19,432 | 19,432 | 3,232 | 3,231 | 21,584 | 9,445 | 12,139 |
| `dev_v1.4.xml` | Development | 2,369 | 2,369 | 439 | 438 | 2,514 | 1,134 | 1,380 |
| `test_TIMESTAMP1.xml` | Test (synchronic) | 2,566 | 2,566 | 471 | 471 | 3,261 | 1,099 | 2,162 |
| `test_TIMESTAMP2.xml` | Test (diachronic) | 1,842 | 1,842 | 295 | 295 | 2,080 | 919 | 1,161 |

### 2.2 Critical dataset anomalies and evaluator traps discovered in audit

Every implementation module must explicitly account for the following verified properties of the raw GermEval 2017 XML and `EvaluationScript.jar`:

1. **Irrelevant documents (`relevance=false`) vs. empty opinions:** Roughly 16–18% of documents in every split have `<relevance>false</relevance>` and zero `<Opinion>` elements (plus 1 relevant document in train and 1 in dev with zero opinions). In Task C and Task D, these are **not** filtered out; they are valid benchmark documents whose gold opinion set is empty (`[]`).
2. **Gold XML typos and taxonomy drift across splits:**
   - **Polarity typos:** `train_v1.4.xml` contains 1 opinion with `polarity="positve"` (missing `i`, doc `682496475092398081`). `test_TIMESTAMP2.xml` contains 1 opinion with `polarity=" negative"` (leading space, doc `803663969349148673`, which also corrupts column 5 of `test_TIMESTAMP2.tsv` into `'Connectivity: negative'`).
   - **Category typos and rare split-specific categories:** The standard GermEval 2017 taxonomy has **19 main aspect categories**. However, `train_v1.4.xml` has 1 instance of `category="Sichaupt"` (a corrupted merge of `Sicherheit#Haupt`, doc `1104163382948573`), and `dev_v1.4.xml` (1 instance) + `test_TIMESTAMP1.xml` (1 instance) contain `category="QR-Code#Haupt"`, which **never appears in `train_v1.4.xml`**.
3. **Discontinuous explicit targets (`*` separator) in ~8–11% of explicit opinions:**
   - Out of 12,139 explicit opinions in `train_v1.4.xml`, **945 do not match `text[from:to] == target`**. Of those, **938 (99.3%) are discontinuous multi-part targets** where the annotators joined separated words with `" * "` in the `target` attribute (e.g., `target="lernt * Bahn wieder nicht"` for text `"lernt die Bahn wieder nicht"`, or `target="zeigt * an"` for `"zeigt sie einfach mal falsche Abfahrtszeiten an"`).
   - In `train_v1.4.xml`, `dev_v1.4.xml`, and `test_TIMESTAMP1.xml`, `from` and `to` span the **outer bounding box** from the start of the first part to the end of the last part (or first part only in a few cases), while `from2="0" to2="0"`.
   - In `test_TIMESTAMP2.xml`, 127 opinions use non-zero `from2`/`to2` for the second component while `from`/`to` covers only the first component!
   - Only 7 opinions in train, 0 in dev, 2 in test1, and 7 in test2 are genuine non-`*` offset/typo mismatches (e.g., trailing space in `target`, or 2 annotations in test2 where `from > 0` and `to = 0`: `target="#Navigator" from="68" to="0"` and `target="DB Navigator" from="63" to="0"`).
4. **Java `EvaluationScript.jar` implementation quirks (`GermevalEvaluator.java`, `OTEMatcher.java`, `Fscore.java`):**
   - **Argument order:** `Evaluate.java` and the `Makefile` invoke `java -jar EvaluationScript.jar <task> <PREDICTED_FILE> <GOLD_FILE>`, even though `GermEval2017/README.md` mistakenly states `<GOLD> <PREDICTION>`. Because micro-F1 is symmetric, overall micro-F1 is unaffected, but precision/recall in custom analyses will invert if swapped. Always pass `<PREDICTED> <GOLD>` to match `GermevalEvaluator.evaluate(predicted, gold, toEvaluate)`.
   - **Task C evaluates TWO metrics:** `evaluateCategory` prints both:
     1. *Categories* (main category only, stripping `#...`, 19 classes)
     2. *Categories + Sentiment* (`<main_category>_<polarity>`, e.g., `Zugfahrt_negative`)
     Both include **all** opinions in a document (both explicit and implicit `NULL` targets).
   - **Task C false-positive bug on empty gold documents (`GermevalEvaluator.java:140`):** In `evaluateCategory`, the Java evaluator checks `if (doc_gold.getAspects() != null && doc_gold.getAspects().getAspects() != null)` before comparing predictions. In XML mode, JAXB unmarshals `<Opinions/>` (or missing `<Opinions>`) such that `getAspects()` is `null` on empty documents; in TSV mode, `GermevalReader.java:76` leaves the list empty (`size == 0`, not `null`), **except** when `line.split("\t")` strips trailing empty columns so `lineParts.length <= 4`! Consequently, in XML evaluation (and TSV evaluation when column 5 is empty), **any false-positive aspect predicted on a gold-empty document (`relevance=false` or 0 opinions) is silently ignored by `EvaluationScript.jar`!** Our Python evaluator (`evaluation.py`) must report both **strict micro-F1** (which penalizes false positives on empty documents) and **legacy Java-compatible micro-F1** (which replicates `EvaluationScript.jar` behavior).
   - **Task D (`OTE`) ignores polarity and ignores `from2`/`to2` (`OTEMatcher.java:54,105`):** The official `OTE` evaluator checks **only** `(begin, end, main_category)`—it completely ignores `polarity` and `from2`/`to2`! Furthermore, in `EvaluationScript.jar`, `OTE` includes implicit opinions `(0, 0, main_category)` if present in the XML! And in `tsv2evalinputxml.py:185`, multi-token targets were historically concatenated without spaces (`lastcontent += currentcontent`), which worked only because `OTEMatcher.java` checks integer `begin`/`end` offsets and ignores the `target` string attribute.

### 2.3 Non-negotiable data rules

- Preserve original text verbatim; never normalize it in place.
- Preserve all XML opinion records, full `category` values (including `#` suffix), polarity, original target string, and all original offset attributes (`from`, `to`, `from2`, `to2`) in **canonical data**.
- Preserve relevance and document sentiment separately from aspect–polarity opinions.
- Keep explicit and implicit opinions distinct (`is_implicit = (target == "NULL" and from == 0 and to == 0)`), while flagging the 2 anomalous `test_TIMESTAMP2` records (`from > 0, to == 0`) in the validation audit.
- Preserve duplicate and overlapping opinions in canonical data; only deduplicate when constructing Task C **label sets**.
- The LLM's prompt and expected completion **must not include character offsets, token indices, or the historical `__D__` token**.
- Preserve original train/dev/test boundaries and report exact duplicates or near-duplicates across splits; do not silently modify the official benchmark.

## 3. Project layout

Use `uv` + `pyproject.toml` for deterministic Python environment management (with platform/extras separation for CUDA training vs. macOS CPU/MLX development):

```text
gemma-qlora-finetuning/
├── README.md
├── finetuning-demo.md
├── pyproject.toml                     # uv project with [project.optional-dependencies] cuda, macos, dev
├── uv.lock
├── Makefile
├── configs/
│   ├── dataset.yaml
│   ├── model.yaml
│   ├── prompts.yaml
│   ├── lora.yaml
│   ├── qlora.yaml
│   └── experiments.yaml
├── data/
│   ├── raw/                           # symlinks or immutable copies of train_v1.4.xml, dev_v1.4.xml, test_TIMESTAMP1/2.xml & .tsv
│   ├── canonical/                     # train/dev/test_syn/test_dia.jsonl
│   ├── derived/
│   │   ├── task_c/                    # conversational prompt/completion JSONL
│   │   └── task_d/                    # conversational prompt/completion JSONL
│   └── manifests/                     # hashes, counts, taxonomy.json, validation_audit.json, patch_table.json
├── src/germeval_llm/
│   ├── __init__.py
│   ├── schemas.py
│   ├── xml_parser.py
│   ├── validate.py
│   ├── convert.py
│   ├── taxonomy.py
│   ├── prompts.py
│   ├── inference.py
│   ├── parsing.py
│   ├── alignment.py
│   ├── evaluation.py
│   ├── official_evaluation.py
│   ├── training.py
│   └── reporting.py
├── notebooks/
│   ├── 00_environment.ipynb
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_xml_conversion.ipynb
│   ├── 03_instruction_dataset.ipynb
│   ├── 04_prompting_baselines.ipynb
│   ├── 05_lora_finetuning.ipynb
│   ├── 06_qlora_finetuning.ipynb
│   ├── 07_learning_curves.ipynb
│   ├── 08_task_d_extraction.ipynb
│   ├── 09_evaluation.ipynb
│   └── 10_error_analysis.ipynb
├── tests/
│   ├── fixtures/
│   ├── test_xml_parser.py
│   ├── test_conversion.py
│   ├── test_task_c.py
│   ├── test_alignment.py
│   └── test_official_evaluation.py
├── results/
│   ├── predictions/
│   ├── metrics/
│   ├── runs/
│   └── figures/
└── models/adapters/                   # ignored by git
```

**Engineering rule:** notebooks orchestrate, visualize, and explain; the `src/germeval_llm/` package owns parsing, conversion, alignment, training configuration, and evaluation. Every notebook must be restartable and use paths relative to the repository root.

## 4. Dataset conversion: canonical representation

### 4.1 Parse XML into one record per document

Use `lxml.etree` (or `xml.etree.ElementTree` with strict UTF-8 handling) and a Pydantic v2 schema (`schemas.py`). An *illustrative* canonical record is:

```json
{
  "doc_id": "example-001",
  "source_split": "train",
  "source_ordinal": 0,
  "text": "Der Zug fällt aus. Der Service ist schlecht.",
  "relevance": true,
  "document_sentiment": "negative",
  "opinions": [
    {
      "category_full": "Zugfahrt#Haupt",
      "category": "Zugfahrt",
      "subcategory": "Haupt",
      "polarity": "negative",
      "target_xml": "Zug",
      "start_xml": 4,
      "end_xml": 7,
      "start2_xml": 0,
      "end2_xml": 0,
      "is_implicit": false,
      "is_discontinuous_star": false,
      "xml_attributes": {"category": "Zugfahrt#Haupt", "target": "Zug", "from": "4", "to": "7", "polarity": "negative"}
    }
  ]
}
```

**Illustrative only:** do not use this example as gold. Preserve additional XML attributes such as `from2`/`to2` (present on 127 records in `test_TIMESTAMP2.xml`); avoid discarding fields that the original XML carries. Keep `source_ordinal` to reconstruct exact document order. Preserve opinion order in canonical JSONL.

### 4.2 Validation, auditing, and explicit patch policy

Implement `validate_canonical(record)` and emit a machine-readable audit file (`data/manifests/validation_audit.json`) with severity, document ID, opinion index, issue code, and relevant values. Checks:

- Required attributes and valid data types.
- Taxonomy and polarity anomalies:
  - Flag `polarity="positve"` (train), `polarity=" negative"` (test2), `category="Sichaupt"` (train), and `category="QR-Code#Haupt"` (dev, test1).
- Explicit target classification (`target_xml != "NULL"`):
  - **Exact continuous match (`EXACT_SUBSTRING`):** `text[start_xml:end_xml] == target_xml` (11,194 in train; 1,282 in dev; 2,043 in test1; 1,030 in test2).
  - **Discontinuous `*` match (`DISCONTINUOUS_STAR`):** `"*" in target_xml` (938 in train; 98 in dev; 117 in test1; 124 in test2). Verify whether every `*`-separated segment in `target_xml` appears in order within `text` (and within `text[start_xml:end_xml]` when `end2_xml == 0`, or across `[start_xml:end_xml]` and `[start2_xml:end2_xml]` in `test_TIMESTAMP2`).
  - **Other offset/target mismatch (`OFFSET_MISMATCH`):** 7 in train, 0 in dev, 2 in test1, 7 in test2, plus the 2 `from > 0, to == 0` anomalies in test2.
- Repeated targets, duplicate annotations, overlapping spans, multiple category–polarity labels for a target, and implicit opinions.
- Number of XML documents/opinions equals number of canonical documents/opinions; report per-split totals.
- Exact-text and ID duplicates across splits; report but do not silently delete.
- Length distribution under the selected Gemma tokenizer, including outliers.

**Canonical vs. Derived Normalization Policy:**
Do **not** automatically alter `data/canonical/*.jsonl`—canonical files must pass a lossless round-trip test against the raw XML. Instead, store an explicit, versioned `configs/dataset.yaml` patch table applied **only when deriving Task C/D training examples** (while keeping raw test files untouched for official Java evaluation):
- Normalize training/dev polarity typos (`"positve" -> "positive"`, `" negative" -> "negative"`) and training category typo (`"Sichaupt" -> "Sicherheit#Haupt"`) in derived training views, logging exact counts.
- Document `QR-Code` (1 in dev, 1 in test1, 0 in train): keep the official 19 main categories as the primary taxonomy (or explicitly document whether `QR-Code` is included as a 20th zero-shot category in `taxonomy.json`).

### 4.3 Manifest and dataset report

Generate `data/manifests/dataset_manifest.json` with source filenames, SHA-256, parser version, git revision, timestamp, split sizes, annotation counts, categories, polarities, and validation counts. Generate `statistics.json` and figures: category counts, category × polarity matrix, opinions per document (including the ~17% empty/irrelevant documents), implicit/explicit/discontinuous proportions, span length, text length, overlapping spans, duplicate texts, and synchronic/diachronic distribution shifts.

**Acceptance gate A:** every source document is accounted for (19,432 train / 2,369 dev / 2,566 test1 / 1,842 test2); all original opinion records and XML attributes are retained; all mismatches are categorized; round-trip tests pass.

## 5. Derive Task C training and evaluation data

### 5.1 Target definition and dual official metrics

For each document, form a **set** of `(main_category, polarity)` pairs from **all** of its opinions (both explicit and implicit `NULL` targets, matching `GermevalEvaluator.getAspectSentimentSet`), stripping the `#Subcategory` suffix. Preserve distinct polarities for the same category in the same document. Documents with zero opinions (including `relevance=false` documents) produce an empty list `[]`.

The compact completion schema is:

```json
{"labels": ["Service_und_Kundenbetreuung:negative", "Zugfahrt:negative"]}
```

Sort labels deterministically by frozen taxonomy order and then polarity order (`positive`, `neutral`, `negative`). Never reduce multilabel annotations to a single class. Freeze the allowed label inventory in `data/manifests/taxonomy.json` (19 main categories × 3 polarities = 57 `category:polarity` labels; document the 1 `QR-Code` outlier in dev/test1 explicitly).

Because the official evaluator (`GermevalEvaluator.evaluateCategory`) reports two Task C scores:
1. **Task C1 (Category-only Micro-F1):** set of unique `main_category` strings per document.
2. **Task C2 (Category + Sentiment Micro-F1):** set of unique `main_category:polarity` pairs per document.

Our single model completion (`{"labels": ["Category:polarity", ...]}`) deterministically yields **both** sets (by projecting `Category:polarity -> Category` for C1). Always report both C1 and C2.

### 5.2 Instruction-tuning examples and Gemma chat template handling

Store conversational prompt/completion examples in `data/derived/task_c/*.jsonl`:

```json
{
  "example_id": "train:000001",
  "doc_id": "http://twitter.com/...",
  "prompt": [
    {
      "role": "user",
      "content": "Klassifiziere den folgenden deutschen Text bezüglich genannter Aspekte der Deutschen Bahn und deren Polarität.\nErlaubte Kategorien: <INSERT VERSIONED TAXONOMY>\nErlaubte Polaritäten: positive, neutral, negative\nGib ausschließlich ein valides JSON-Objekt im Format {\"labels\": [\"Kategorie:Polarität\", ...]} zurück. Falls keine relevanten Aspekte vorliegen, gib {\"labels\": []} zurück.\n\nText:\nDer Zug fällt aus."
    }
  ],
  "completion": [
    {
      "role": "assistant",
      "content": "{\"labels\":[\"Zugfahrt:negative\"]}"
    }
  ]
}
```

**Gemma-specific chat template rule:**
Gemma 2 and Gemma 3 Instruct chat templates only support alternating `user` and `model` (`assistant`) turns and **raise an error if a separate `system` role message is passed** (or require merging system instructions into the first `user` turn). Therefore:
- Place the system/task instructions directly at the top of the `user` message (or provide a deterministic pre-processor in `prompts.py` that merges `system` + `user` before calling `tokenizer.apply_chat_template`).
- Use the **exact same instruction text and taxonomy descriptions** in P1/P2/P3 prompting and in LoRA/QLoRA fine-tuning so that adaptation is not confounded with prompt information.
- **Completion-only loss masking:** In TRL `SFTTrainer` (v0.12+), verify whether native prompt-completion dataset formatting (`{"prompt": [...], "completion": [...]}`) or `DataCollatorForCompletionOnlyLM(response_template="<start_of_turn>model\n", ...)` is used, and write a mandatory unit test confirming that `labels` is `-100` for every token up to and including `<start_of_turn>model\n` and valid token IDs for the JSON completion and `<end_of_turn>`.

### 5.3 Output parser and validation

Implement one parser (`parsing.py`) shared by every model:
1. Extract JSON (supporting both raw JSON output and optional markdown ```json fences in zero-shot baselines, recording whether fence stripping was needed).
2. Require a dict with key `"labels"` whose value is a `list` of strings.
3. Validate each item against the frozen `Category:polarity` inventory.
4. Deduplicate exact duplicates and sort canonically.
5. Save **raw output**, parsed output, parse status (`VALID_EXACT`, `VALID_FENCED`, `INVALID_JSON`, `INVALID_SCHEMA`, `UNKNOWN_LABEL`), and validation errors.

For primary scoring, malformed JSON or unknown labels are rejected (counted as empty/dropped predictions) rather than heuristically fuzzy-matched into gold labels; report strict schema validity rate, unknown-label rate, empty-prediction rate (`{"labels": []}`), and token length separately.

**Acceptance gate B:** deterministic conversion, stable taxonomy, valid JSONL, verified `-100` prompt loss masking on Gemma's `<start_of_turn>model\n` boundary, no evaluation leakage, and unit tests for zero/multiple/conflicting labels.

## 6. Derive Task D: extract target texts, never offsets

### 6.1 Training target and discontinuous expression policy

The model input is the **original full text**, category definitions, and extraction instructions. The model output is a list of opinion target expressions copied from that text, with main category and polarity:

```json
{
  "targets": [
    {"target": "Zug", "category": "Zugfahrt", "polarity": "negative"},
    {"target": "Service", "category": "Service_und_Kundenbetreuung", "polarity": "negative"}
  ]
}
```

**No `start`, `end`, `offset`, `token_index`, `__D__`, or similar fields in the model prompt or target.** This rule applies equally to zero-shot, few-shot, LoRA, QLoRA, and joint Task C+D variants.

**Handling implicit and discontinuous targets in Task D:**
1. **Implicit opinions (`target == "NULL"`, `to == 0`):** Excluded from Task D target extraction (since they have no surface target phrase). Note and document a critical discovery from `OTEMatcher.java`: the historical Java `OTE` evaluator does **not** filter out `(0,0)` opinions from the gold XML if they are present! Therefore, in `official_evaluation.py` for Task D, report:
   - **Explicit-only OTE F1** (evaluating against explicit gold targets, which is the true target-extraction task), and
   - **All-opinions XML OTE F1** (if comparing against historical `Makefile` runs where `__D__` predictions were converted into `from="0" to="0" target="NULL"` XML elements).
   - Also examine the <strong>Joint Task C+D (`J1`)</strong> extension, where outputting `target: null` for implicit document-level opinions naturally unifies Task C and Task D in a single generative schema!
2. **Discontinuous targets (`"*" in target_xml`, ~7.7% of explicit opinions in train):**
   Because German syntax frequently splits verbal brackets and multi-word expressions (e.g., `"zeigt sie einfach mal falsche Abfahrtszeiten an"` annotated as `target="zeigt * an"`, `from="30" to="77"`), define two explicit, testable Task D views in `convert.py`:
   - **Primary view (`exact_And_star`):** Preserve the GermEval `" * "` convention in training and prompting (instructing the model that discontinuous target phrases may be joined with `" * "`, or evaluating how well fine-tuning learns this annotation convention vs. prompting).
   - **Strict contiguous view (`contiguous_only` ablation):** Filter training targets to exact contiguous substrings (`text[from:to] == target`, 92.2% of explicit opinions) to measure pure substring copying without `*` syntax.

### 6.2 Deterministic postprocessing into official spans (`alignment.py`)

To produce `(from, to)` character offsets for official evaluation without the LLM ever seeing offsets:

1. **Contiguous target string (`"*" not in target`):**
   - Find all exact occurrences of `target` in the unchanged source `text`.
   - If **1 occurrence** exists: assign `(start, end)`.
   - If **0 occurrences** exist: record `unaligned` and do not fabricate offsets.
   - If **>1 occurrences** exist (occurs in ~16.6% of explicit gold opinions in train): record `ambiguous`, and apply a deterministic **leftmost-unused span assignment** policy (tracking `(start, end, category)` already assigned in the document so repeated predictions of the same string advance to the next occurrence). Never use gold offsets to resolve ambiguity.
2. **Discontinuous target string (`"*" in target`):**
   - Split on `" * "` into parts $[p_1, p_2, \dots, p_k]$.
   - Find ordered non-overlapping occurrences $s_1 < e_1 \le s_2 < e_2 \dots \le s_k < e_k$ in `text` (preferring minimal total span length $e_k - s_1$ if multiple matches exist).
   - To match the official GermEval XML convention (and `OTEMatcher.java`, which only inspects `from` and `to`), set the bounding span `from = s_1, to = e_k` (and log `s_1..e_1, s_k..e_k`).
3. Preserve separate `category`/`polarity` labels even when they map to the same character span.
4. Keep any optional whitespace/punctuation trimming alignment outside the primary exact-match condition and report its effect as an ablation.

### 6.3 Complementary Task D metrics

- **String-level extraction F1 (Primary model metric):** compare multisets of `(target_string, category, polarity)` (and category-only `(target_string, category)`) per document. This directly measures what the LLM was trained/prompted to generate, independent of offset alignment.
- **Span-level Python F1 (Explicit OTE exact & overlap):** replicate `OTEMatcher.java` exact `(from, to, category)` and $\pm 1$-token overlap matching in Python on explicit gold targets, penalizing unaligned predictions as false positives rather than silently dropping them.
- **Historical `EvaluationScript.jar OTE` score:** run the Java evaluator on the exported XML and report alignment rates (`unique_aligned`, `ambiguous_aligned`, `discontinuous_aligned`, `unaligned`).

**Acceptance gate C:** no offset fields anywhere in model-facing Task D data; both contiguous and `*`-discontinuous targets handled deterministically; repeated-string and zero-match unit tests pass; official XML output validates.

## 7. Notebook-by-notebook implementation

Each notebook includes a short conceptual introduction, configuration display, small runnable demonstration, assertions, saved outputs, visualizations, and a final 'what did we learn?' section. All core logic lives in `src/germeval_llm/`.

### `00_environment.ipynb` — environment, Gemma model check, reproducibility

1. Check Python, PyTorch, CUDA/MPS/CPU, GPU memory, bf16 capability, and installed `transformers`, `datasets`, `trl`, `peft`, `accelerate`, `bitsandbytes`, `scikit-learn`, `lxml`, `pydantic` versions.
2. Verify Hugging Face authentication (`HF_TOKEN` for gated Gemma checkpoint access) using safe credential checks; confirm that the selected Gemma Instruct model and tokenizer load cleanly.
3. Inspect Gemma's chat template (`<bos><start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n...<end_of_turn>`) and verify single-BOS behavior.
4. Set seeds (Python/NumPy/Torch); record hardware and `uv.lock` hash.
5. Run one minimal deterministic inference; inspect input/output token counts.

**Output:** `results/runs/environment.json`. Note: 4-bit bitsandbytes QLoRA training requires a CUDA GPU (e.g., T4/L4/A10/A100/RTX); macOS Apple Silicon users can run dataset preparation, evaluation, and CPU/MPS inference locally, or run the training notebooks on a remote CUDA kernel/Colab/Lightning/cluster.

### `01_dataset_exploration.ipynb` — XML audit and task feasibility

1. Load all four original XML splits (`train_v1.4`, `dev_v1.4`, `test_TIMESTAMP1`, `test_TIMESTAMP2`) read-only.
2. Show real annotated examples: irrelevant (`relevance=false`) documents, implicit (`NULL`) targets, discontinuous (`*`) targets, overlapping labels, multi-label spans, and the discovered XML typos (`positve`, ` negative`, `Sichaupt`, `QR-Code`).
3. Plot category and polarity frequencies and annotation multiplicities across all 4 splits.
4. Quantify Task D feasibility: exact contiguous substring rate (~92.2%), discontinuous `*` rate (~7.7%), non-`*` mismatch rate (<0.1%), and repeated substring ambiguity rate (~16.6%).
5. Verify reconciliation between `test_TIMESTAMP1/2.xml` and `test_TIMESTAMP1/2.tsv`.

**Output:** `data/manifests/statistics.json`, `results/figures/dataset_*.png`.

### `02_xml_conversion.ipynb` — canonical JSONL

1. Demonstrate XML → Pydantic `CanonicalDocument` on representative documents.
2. Convert all 4 splits with `xml_parser.py`.
3. Run `validate_canonical` and display the structured anomaly summary table.
4. Demonstrate lossless round-trip verification against the source XML AST.
5. Save `data/canonical/*.jsonl`, `dataset_manifest.json`, and `validation_audit.json`.

**Output:** `data/canonical/*.jsonl`, audit logs. **Pass criterion:** exact document/opinion counts match Section 2.1 and round-trip tests pass.

### `03_instruction_dataset.ipynb` — Task C and D views

1. Apply the documented training-split typo patch (`positve -> positive`, `Sichaupt -> Sicherheit#Haupt`) and freeze the 19-category `taxonomy.json`.
2. Convert canonical opinions to sorted Task C label sets (`{"labels": [...]}`), verifying that `relevance=false` documents map to `{"labels": []}`.
3. Create Task C and Task D conversational prompt/completion JSONL formatted for Gemma's `user`/`assistant` turn structure.
4. Validate that no offset fields appear anywhere in model-facing Task D examples.
5. Inspect tokenized sequences: verify no double `<bos>` token, inspect max sequence length distribution (99.9th percentile + few-shot headroom), and assert that completion-only loss masking sets `labels = -100` on all prompt tokens.

**Output:** `data/derived/task_c/*.jsonl`, `data/derived/task_d/*.jsonl`, `data/manifests/taxonomy.json`.

### `04_prompting_baselines.ipynb` — P0/P1/P2/P3

1. Load the frozen Gemma Instruct checkpoint and evaluation split.
2. **P0:** minimal German instruction + JSON output contract + list of 19 allowed category names and 3 polarities.
3. **P1:** add official GermEval 2017 category descriptions and annotation conventions (explicit + implicit aspects, empty list for irrelevant posts).
4. **P2:** add deterministic 2, 5, and 10 training-set demonstrations (stratified to include relevant multi-label, implicit, and irrelevant `[]` examples).
5. **P3:** optimize prompt wording/demonstration selection **only on `dev_v1.4`**; optionally compare unconstrained greedy decoding (`do_sample=False`) against schema-constrained decoding as an explicit sub-experiment.
6. Save raw and parsed outputs, Task C1 (Category) and C2 (Category+Sentiment) micro/macro-F1, schema validity rates, prompt/completion token counts, and latency.

**Output:** `results/predictions/prompt_*.jsonl`, `results/metrics/prompt_*.json`.

### `05_lora_finetuning.ipynb` — bf16/fp16 adapter SFT

1. Load the same Gemma Instruct checkpoint in native `bfloat16` (or `float16` if hardware requires, recording dtype).
2. Explain low-rank adaptation ($W + \frac{\alpha}{r} BA$) and print trainable vs. total parameters.
3. Apply PEFT LoRA to all 7 linear projection layers of Gemma (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
4. Train on Task C with completion-only loss, `paged_adamw_8bit` or `adamw_torch` (matched with Notebook 06), cosine LR schedule, `2e-4` learning rate, effective batch size 16, for 3 epochs.
5. Evaluate on `dev_v1.4` after each epoch using exact JSON generation + `evaluation.py` (selecting best checkpoint by dev Task C2 micro-F1, not validation loss alone).
6. Record training wall time, peak allocated/reserved GPU VRAM, adapter byte size, and dev/test metrics.

**Output:** LoRA adapter checkpoint, training log, predictions, metrics, memory profile.

### `06_qlora_finetuning.ipynb` — 4-bit NF4 QLoRA (central CAS lab)

1. Load the same Gemma Instruct checkpoint with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)`.
2. Apply `prepare_model_for_kbit_training` and attach the exact same LoRA configuration (`r=16, lora_alpha=32, lora_dropout=0.05`, same 7 target modules) as Notebook 05.
3. Run a fast 100-step classroom smoke test, followed by the full training run on identical data, batch size, and seed as LoRA.
4. Save the adapter, reload it onto a fresh 4-bit base model, and verify deterministic prediction reproducibility.
5. Compare VRAM footprint, throughput, adapter size, and Task C1/C2 F1 directly against 16-bit LoRA and P0–P3 prompting.

**Output:** QLoRA adapter, predictions, LoRA-vs-QLoRA memory/accuracy comparison table.

### `07_learning_curves.ipynb` — data efficiency and training budget dynamics

1. Construct nested training subsets of $N \in \{100, 250, 500, 1000, 2000, 4000, \text{all (19,432)}\}$ documents.
2. Compare uniform random sampling (which naturally includes ~17% empty/irrelevant docs) against iterative multilabel stratified sampling (`iterative-stratification`) that guarantees rare-category representation at small $N$.
3. **Crucial methodological control:** At $N=100$ with batch size 16, 3 epochs is only 18 optimizer steps (barely past warmup!). Define a **minimum step floor** (e.g., $\max(3 \text{ epochs}, 100 \text{ steps})$ with early stopping on dev) so small-$N$ points are not artificially crippled by under-optimization, and document both step counts and epoch counts.
4. Plot Task C1/C2 micro-F1, macro-F1, rare-category recall, and GPU training time across $N$ (with 3 seeds for precomputed curves), overlaid against horizontal P0–P3 prompting baselines and their per-1,000-document inference token costs.

**Output:** learning-curve CSV/JSON and publication-quality figures.

### `08_task_d_extraction.ipynb` — target-string generation and span alignment

1. Display original text and gold Task D target strings (highlighting contiguous targets vs. `" * "` discontinuous targets); verify no offsets appear in prompts.
2. Evaluate zero-shot/few-shot prompting (`D1`) and QLoRA fine-tuning (`D2`) on Task D target-string generation.
3. Optionally evaluate Joint Task C+D (`J1`), where implicit opinions are emitted with `"target": null` and explicit opinions with verbatim `"target": "..."`.
4. Run `alignment.py` as a separate deterministic post-processing stage, visualizing unique matches, leftmost-unused disambiguations, discontinuous `*` bounding spans, and unaligned predictions.
5. Report (a) String-level multiset F1, (b) Python explicit-span exact & overlap F1, and (c) official `EvaluationScript.jar OTE` scores.

**Output:** Task D predictions, alignment audit breakdown, string vs. span F1 metrics.

### `09_evaluation.ipynb` — authoritative benchmark & official Java reconciliation

1. Load saved predictions from all runs (`P0–P3`, `L1`, `Q1–Q3`, `D1–D2`, `J1`) across `dev_v1.4`, `test_TIMESTAMP1` (synchronic), and `test_TIMESTAMP2` (diachronic).
2. Compute Python Task C metrics: strict C1/C2 micro-F1, macro-F1, exact-set accuracy, per-category precision/recall/F1, and paired bootstrap 95% confidence intervals ($B=1000$).
3. Export predictions to official TSV/XML formats (`official_evaluation.py`) and invoke `java -jar EvaluationScript.jar category <PRED> <GOLD>` and `OTE <PRED> <GOLD>`.
4. Demonstrate exact numerical reconciliation between our Python legacy-mode scorer and `EvaluationScript.jar` (explicitly showing the impact of the Java evaluator's empty-gold-document bug and the `test_TIMESTAMP2` leading-space polarity typo).

**Output:** `results/metrics/benchmark.csv`, reconciliation verification table.

### `10_error_analysis.ipynb` — linguistic and systems interpretation

1. Plot per-category F1 and $\Delta\text{F1} = \text{F1}_{\text{QLoRA}} - \text{F1}_{\text{P3}}$ against training category support (log-scale).
2. Inspect paired document quadrants: both correct, prompt-only correct, QLoRA-only correct, neither correct.
3. Analyze false positives on `relevance=false` documents (do LLMs over-predict `Allgemein` or `Image` on irrelevant railway news/tweets?), implicit vs. explicit opinion recall, polarity confusions (`neutral` vs. `negative`), and JSON schema failures.
4. Compare synchronic (`test_TIMESTAMP1`) vs. diachronic (`test_TIMESTAMP2`) generalization drop.
5. For Task D, decompose total span error into (a) LLM extraction error, (b) discontinuous `*` boundary mismatch, and (c) repeated-string offset ambiguity.

**Output:** error-analysis figures and concise pedagogical synthesis.

## 8. Model selection and training configuration

Use a single gated/public **Google Gemma Instruct** checkpoint across all experiments so prompting, LoRA, and QLoRA share the exact same base weights and tokenizer:
- **Primary candidate:** `google/gemma-2-2b-it` (2.6B parameters; fast, fits comfortably on any 16 GB T4/L4/RTX GPU for both 16-bit LoRA and 4-bit QLoRA, strong German instruction following) or `google/gemma-3-4b-it` (4B parameters; 128K context, strong multilingual performance).
- Freeze the exact Hugging Face Hub model ID and commit SHA in `configs/model.yaml` before running test evaluations.

Baseline training hyperparameters (`configs/lora.yaml` and `configs/qlora.yaml`):

| Parameter | Initial setting | Notes / Gemma specifics |
|---|---|---|
| Base model | `google/gemma-2-2b-it` (or `gemma-3-4b-it`) | Pin exact commit revision SHA |
| LoRA rank ($r$) | 16 | Ablate $r \in \{8, 16, 32\}$ in `Q3` |
| LoRA alpha ($\alpha$) | 32 | Scaling $\alpha/r = 2.0$ |
| LoRA dropout | 0.05 | Identical for LoRA and QLoRA |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | All 7 attention + SwiGLU MLP linear projections |
| QLoRA quantization | 4-bit NF4, double quant, compute `bfloat16` | Fallback to `float16` compute only on pre-Ampere GPUs (e.g., T4) and log it |
| Attention implementation | `sdpa` or `eager` (or `flash_attention_2` on Ampere+) | Note: Gemma 2 uses soft-capping; `sdpa`/`eager` in recent `transformers` handles this accurately |
| Optimizer | `paged_adamw_8bit` (or `adamw_torch`) | Keep identical between LoRA and QLoRA to isolate quantization effect |
| Learning rate | `2e-4` with cosine schedule, 5% warmup | Tune on `dev_v1.4` only |
| Training budget | 3 epochs on full train ($\ge 100$ steps floor on small subsets) | Select best checkpoint by dev Task C2 micro-F1 |
| Effective batch size | 16 (e.g., `per_device_train_batch_size=4`, `gradient_accumulation_steps=4`) | Disable sample packing so completion-only masking is 1-to-1 per document |
| Max sequence length | Data-derived (e.g., 1024 for SFT; 2048 for 10-shot prompting) | Verify 0% completion truncation across all splits |
| Generation | Greedy (`do_sample=False`), `max_new_tokens=256` | Identical across prompting, LoRA, and QLoRA |
| Seeds | `42` for single runs; `{42, 100, 2026}` for learning curves | Record all seeds and variance |

## 9. Experiment matrix

| ID | Approach | Task | Required? | Primary comparison |
|---|---|---|---|---|
| `P0` | Minimal zero-shot | C | Yes | Out-of-the-box Gemma baseline |
| `P1` | Guideline/taxonomy zero-shot | C | Yes | Value of domain instructions |
| `P2` | Fixed few-shot (2, 5, 10 shots) | C | Yes | In-context learning vs. token cost |
| `P3` | Dev-optimized few-shot | C | Yes | Strongest non-finetuned baseline |
| `L1` | 16-bit LoRA ($r=16$) | C | Yes | Standard adapter fine-tuning |
| `Q1` | 4-bit QLoRA ($r=16$) | C | Yes | Memory-efficient adapter fine-tuning |
| `Q2` | QLoRA learning curve ($N=100\dots\text{all}$, random vs. stratified) | C | Yes (precomputed) | Data efficiency & crossover vs. `P3` |
| `Q3` | QLoRA rank ablation ($r \in \{8, 16, 32\}$) | C | Optional | Adapter capacity sensitivity |
| `D1` | Prompting (`P3` style), target strings only | D | Extension | Generative OTE baseline |
| `D2` | QLoRA ($r=16$), target strings only | D | Extension | Fine-tuned OTE + span alignment |
| `J1` | Joint Task C+D QLoRA (`target: null` for implicit, string for explicit) | C+D | Optional | Unified aspect-sentiment-target extraction |

## 10. Evaluation protocol and leakage prevention

1. **Training (`train_v1.4`):** Fit adapters and sample few-shot demonstrations strictly from `train_v1.4`.
2. **Development (`dev_v1.4`):** Tune prompt wording, select few-shot examples, choose learning rate/epochs, and pick best checkpoints on `dev_v1.4`. Log all dev evaluations.
3. **Test (`test_TIMESTAMP1` synchronic & `test_TIMESTAMP2` diachronic):** Evaluate frozen configurations once on both official test splits.
4. **Task C metrics:** Report both **Task C1 (Category Micro-F1)** and **Task C2 (Category+Sentiment Micro-F1)** under (a) **Strict Python scoring** (penalizing false positives on empty/irrelevant documents) and (b) **Official Java-compatible scoring** (`EvaluationScript.jar`), plus Macro-F1, Exact-Set Accuracy, Schema Validity Rate, and paired bootstrap 95% CIs.
5. **Task D metrics:** Report (a) **String-level multiset F1** (primary model behavior metric), (b) **Explicit-target Span F1** (exact and $\pm 1$-token overlap in Python), and (c) **Official `EvaluationScript.jar OTE` F1**, alongside alignment breakdown (`unique`, `ambiguous`, `discontinuous`, `unaligned`).
6. **Efficiency metrics:** Prompt/completion tokens per 1,000 docs, wall-clock inference latency, peak GPU VRAM (allocated & reserved), training GPU-minutes, and adapter disk size.
7. **Data licensing & privacy:** GermEval 2017 data is licensed under CC BY-NC 2.0; respect non-commercial attribution terms and avoid committing raw social media text screenshots outside local repository guidelines.

## 11. Required unit and integration tests (`tests/`)

- **`test_xml_parser.py` & `test_conversion.py`:**
  - Document with 0 opinions and `relevance=false`; document with 0 opinions and `relevance=true`.
  - Implicit `NULL` target (`from=0, to=0`); anomalous `from > 0, to=0` target (`test_TIMESTAMP2`).
  - Discontinuous `*` target (`train_v1.4` with `from2=0, to2=0` and `test_TIMESTAMP2` with `from2 > 0, to2 > 0`).
  - Typos (`positve`, ` negative`, `Sichaupt`, `QR-Code`) flagged in audit and handled per patch policy.
  - Lossless XML-to-canonical round-trip verification.
- **`test_task_c.py`:**
  - Duplicate `(category, polarity)` pairs deduplicated; conflicting polarities for the same category retained.
  - Both C1 (`category`) and C2 (`category:polarity`) sets derived accurately.
  - Parser rejects unknown categories/polarities, handles markdown-fenced vs. raw JSON, and preserves empty predictions `{"labels": []}`.
  - Gemma chat template formatting produces a single `<bos>` and masks 100% of prompt tokens with `-100` up to `<start_of_turn>model\n`.
- **`test_alignment.py` (Task D):**
  - Unique contiguous substring $\rightarrow$ exact `(start, end)`.
  - Repeated substring in document $\rightarrow$ flagged `ambiguous`, deterministic leftmost-unused span assigned.
  - Discontinuous `"A * B"` target $\rightarrow$ ordered span search returns bounding `(start_A, end_B)`.
  - Hallucinated/missing substring $\rightarrow$ flagged `unaligned`, penalized in Python scorer, excluded from XML export without crashing.
- **`test_official_evaluation.py`:**
  - Verify argument order (`<PREDICTED> <GOLD>`) against `EvaluationScript.jar`.
  - Verify 100.0% F1 when gold XML/TSV is evaluated against itself.
  - Verify exact numerical agreement between `evaluation.py` (in legacy Java-compatible mode) and `EvaluationScript.jar` on fixture predictions covering empty docs, multi-label docs, and OTE spans.

## 12. Experiment metadata

Every run writes a JSON manifest (`results/runs/<run_id>.json`) alongside its predictions and metrics:

```json
{
  "run_id": "task_c_qlora_r16_seed42",
  "task": "C",
  "method": "qlora",
  "base_model_id": "google/gemma-2-2b-it",
  "base_model_revision": "PINNED_COMMIT_SHA",
  "dataset_manifest_sha256": "COMPUTED_SHA256",
  "prompt_version": "v1",
  "taxonomy_version": "v1_19cats",
  "seed": 42,
  "train_examples": 19432,
  "optimizer_steps": 3643,
  "lora_rank": 16,
  "lora_alpha": 32,
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
  "learning_rate": 0.0002,
  "epochs": 3,
  "quantization": "nf4_double_bf16",
  "checkpoint_path": "models/adapters/task_c_qlora_r16_seed42",
  "metrics": {
    "dev": {},
    "test_syn": {},
    "test_dia": {}
  },
  "efficiency": {
    "peak_vram_mb": 0,
    "train_wall_sec": 0,
    "eval_tokens_per_doc": 0
  },
  "environment": {}
}
```

Also save a JSONL prediction file per split (`results/predictions/<run_id>_<split>.jsonl`) containing `doc_id`, `source_ordinal`, `split`, `raw_output`, `parsed_output`, `parse_status`, `prompt_tokens`, `completion_tokens`, `latency_ms`, and `run_id`. Never overwrite completed runs silently.

## 13. CAS classroom design (90–120 minutes)

**Before class:** pin `uv.lock`; cache the Gemma Instruct checkpoint and pre-converted `data/canonical/` + `data/derived/` files; precompute `P0–P3`, `L1`, `Q1`, `Q2` learning curves, and `D1–D2` predictions; pretrain and verify adapters so live classroom execution never depends on multi-hour GPU jobs.

| Time | Activity | Notebooks | Student output |
|---|---|---|---|
| 0–15 min | Explore real German railway posts, taxonomy, implicit & discontinuous `*` annotations | `01`, `03` | Annotation hypotheses & schema check |
| 15–35 min | Compare minimal (`P0`), guideline (`P1`), and few-shot (`P2/P3`) prompting on Gemma | `04` | Baseline F1, token cost, & failure cases |
| 35–55 min | Inspect 4-bit NF4 quantization, Gemma linear layers, LoRA rank/alpha, and `-100` loss masking | `05`, `06` | VRAM & trainable parameter analysis |
| 55–85 min | Run short 100-step QLoRA smoke training live AND load full 3-epoch pretrained QLoRA adapter | `06` | Live training curve & adapter predictions |
| 85–105 min | Evaluate prompting vs. LoRA vs. QLoRA across synchronic/diachronic splits & learning curves | `07`, `09` | C1/C2 Micro/Macro-F1 & crossover analysis |
| 105–120 min | Error analysis & Task D extension (verbatim string extraction $\rightarrow$ deterministic span alignment) | `08`, `10` | Evidence-based synthesis of when to fine-tune |

**Fallback:** if live CUDA GPUs are unavailable in the classroom, students run all data inspection, prompt/completion tokenization, output parsing, deterministic Task D span alignment, and Python/Java evaluation locally on CPU/macOS using the precomputed prediction artifacts and adapters.

## 14. Milestones and acceptance criteria

### M1 — Source audit, schemas, and canonical conversion (1 day)
- Link or copy the 4 XML and 2 TSV files from `konvens-2018-german-absa` into `data/raw/`.
- Implement `schemas.py`, `xml_parser.py`, `validate.py`, and unit tests.
- Generate `data/canonical/*.jsonl`, `dataset_manifest.json`, and `validation_audit.json` (verifying all counts and anomalies from Section 2.1–2.2).

### M2 — Task C & D instruction datasets + evaluation harness (1 day)
- Implement `taxonomy.py`, `convert.py`, `prompts.py`, `parsing.py`, `evaluation.py`, and `official_evaluation.py`.
- Verify exact reconciliation between `evaluation.py` (Java-compatible mode) and `GermEval2017/EvaluationScript.jar` on gold self-evaluation and synthetic fixtures.
- Generate Gemma-formatted Task C and Task D JSONL datasets and verify `-100` completion-only loss masking.

### M3 — Strong prompting baselines (`P0–P3`) (1–2 days)
- Implement batched/cached inference in `inference.py`.
- Run `P0`, `P1`, `P2` (2/5/10-shot), and `P3` on `dev_v1.4`, freeze the winning prompt configuration, and evaluate on `test_TIMESTAMP1` and `test_TIMESTAMP2`.

### M4 — LoRA (`L1`) and QLoRA (`Q1`) fine-tuning (2 days + GPU runtime)
- Implement `training.py` using `trl.SFTTrainer` and `peft` targeting all 7 Gemma linear modules.
- Train and evaluate `Q1` (4-bit NF4 QLoRA) and `L1` (16-bit LoRA), recording VRAM, runtime, and C1/C2 metrics across dev and both test splits.

### M5 — Learning curves (`Q2`) and rank ablation (`Q3`) (1–2 days + GPU runtime)
- Implement stratified vs. uniform subset sampling with the minimum optimizer step floor ($\ge 100$ steps).
- Run `Q2` across subset sizes and seeds; generate learning curve plots comparing QLoRA against `P0–P3`.

### M6 — Task D extension (`D1`, `D2`, `J1`) (2 days)
- Implement `alignment.py` supporting contiguous substrings, leftmost-unused disambiguation, and `" * "` discontinuous bounding spans.
- Run prompting (`D1`) and QLoRA (`D2`) on Task D, reporting string-level multiset F1, Python explicit-span F1, and official `EvaluationScript.jar OTE` scores.

### M7 — Notebooks `00–10` and classroom release (1–2 days)
- Populate all 11 notebooks cleanly calling `src/germeval_llm/`, verify end-to-end execution, and finalize summary figures and instructor notes.

## 15. Implementation status, technical findings, and live verification

All milestones (`M1–M7`) have been implemented and verified in this repository:

### 15.1 Key technical discoveries during implementation
1. **Official `EvaluationScript.jar` `Fscore.java` Jaccard (`TP / (TP + FP + FN)`) discrepancy:**
   - Decompiling `de.germeval2017.Evaluation.EvaluationUtil.Fscore.java` (lines 37–69) revealed that `getDistinctLabels(data)` collects distinct values across both the `gold` and `predicted` columns of its internal pair register—including the dummy mismatch placeholders `"none"`, `"none_sentiment"`, and `"missmatch"` registered via `register(gold, "none")` (for false negatives) and `register("none", pred)` (for false positives).
   - Because `"none"` never matches itself (`tp=0`), iterating over `"none"` adds every false positive to `fn_sum` and every false negative to `fp_sum`, doubling `fp_sum + fn_sum` in the global accumulator. Consequently, `EvaluationScript.jar`'s reported `MICRO_F1` equals $\frac{2\text{TP}}{2\text{TP} + 2(\text{FP} + \text{FN})} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$ (the Jaccard index) rather than standard Micro-F1 $\frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$.
   - `src/germeval_llm/evaluation.py` implements both **Strict Standard Micro-F1** (`compute_set_classification_metrics`) and **Exact Java-Compatible Micro-F1** (`compute_java_fscore_micro_f1`), reconciled to 6 decimal places against `EvaluationScript.jar` in `tests/test_official_evaluation.py`.
2. **Java 8 JAXB requirement for `EvaluationScript.jar OTE`:**
   - The official `OTE` XML evaluator uses `javax.xml.bind.JAXBContext`, which was removed from the standard JDK in Java 11+. `src/germeval_llm/official_evaluation.py` automatically detects and invokes a local Java 8 runtime (`/Library/Internet Plug-Ins/JavaAppletPlugin.plugin/Contents/Home/bin/java`) when running `OTE` mode while remaining compatible with Java 8+ for `category` TSV mode.
3. **TRL `1.16.0` & Transformers `5.20.1` Completion-Only Loss Masking:**
   - In `trl>=1.0`, `DataCollatorForCompletionOnlyLM` was replaced by native prompt-completion masking via `SFTConfig(completion_only_loss=True)` over datasets structured with explicit `"prompt"` and `"completion"` columns.
   - `src/germeval_llm/training.py` formats each example into `prompt` (`<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n`) and `completion` (`{"labels":[...]}<end_of_turn>\n`) columns so `SFTTrainer` masks 100% of prompt tokens with `-100` while computing cross-entropy loss strictly on the JSON completion tokens.

### 15.2 Live `google/gemma-2-2b-it` smoke verification on Apple Silicon (`mps`)
We verified live inference and LoRA SFT using `google/gemma-2-2b-it` on Apple M4 Pro (48 GB unified memory, PyTorch `2.14.1` `mps` backend):
- **Zero-shot `P1` smoke inference (`smoke_p1_dev`, 5 dev docs):**
  - Raw out-of-the-box `google/gemma-2-2b-it` wrapped outputs in markdown ```` ```json ... ``` ```` fences on 3/5 documents (`VALID_FENCED`: 60%, `VALID_EXACT`: 0%) and invented nested German keys (`{"Kategorie": "...", "Polarität": "..."}`) on 2/5 documents (`INVALID_SCHEMA`: 40%), with mean latency `2,197 ms/doc`.
- **5-step LoRA SFT smoke run (`smoke_task_c_lora`, `subset_size=64`, `max_steps=5`, $r=16, \alpha=32$):**
  - Verified exact parameter counts on live model: `20,766,720` trainable parameters out of `2,635,108,608` total (`0.7881%`), matching our analytical formula in `estimate_gemma_parameter_and_memory_profile()`.
  - Completion-only training loss dropped across the 5 gradient steps from `3.091` $\rightarrow$ `3.228` $\rightarrow$ `1.081` $\rightarrow$ `0.8601` $\rightarrow$ `0.5528` (token accuracy increased from `54.0%` to `85.4%`).
- **Reloaded LoRA adapter inference (`smoke_lora_dev`, same 5 dev docs):**
  - Even after just 5 optimizer steps, reloading the PEFT adapter eliminated markdown fences and nested German dictionary keys on 4/5 documents, raising `VALID_EXACT` raw JSON compliance from **0% to 80%** (`{"labels":["Zugfahrt:negative"]}`) and halving mean completion latency on valid outputs to `~1,100 ms/doc`.

## 16. References and implementation starting points

- Historical implementation and local data source: https://github.com/simon-clematide/konvens-2018-german-absa (`konvens-2018-german-absa`)
- GermEval 2017 shared task website: https://sites.google.com/view/germeval2017-absa/data
- Wojatzki et al. (2017), *GermEval 2017: Shared Task on Aspect-based Sentiment in Social Media Customer Feedback*
- Google Gemma 2 / Gemma 3 documentation: https://huggingface.co/docs/transformers/main/en/model_doc/gemma2
- Hugging Face TRL `SFTTrainer`: https://huggingface.co/docs/trl/sft_trainer
- Hugging Face PEFT LoRA: https://huggingface.co/docs/peft/main/en/conceptual_guides/lora
- Hugging Face Transformers bitsandbytes quantization: https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes

**Final design principle:** The LLM learns the **linguistic annotation** (aspect categories, polarity, and—when doing Task D—verbatim target expressions). The deterministic software layer handles XML conversion, data auditing, duplicate policy, canonical sorting, and any character offsets required by legacy evaluation. The CAS demonstration tests the benefit of LoRA/QLoRA adaptation empirically against strong prompting baselines on the same Gemma Instruct checkpoint.


