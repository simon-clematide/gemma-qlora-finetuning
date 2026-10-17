"""Generate and execute all 12 CAS demonstration notebooks (00_environment to 11_colab_l4_live_demo)
with comprehensive pedagogical explanations, mathematical derivations, inline visualizations,
pure-Python evaluation demos, and interactive student exercises.
"""

from __future__ import annotations

from pathlib import Path
import nbclient
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


SETUP_SNIPPET = """from pathlib import Path
import os, sys, json
from IPython.display import Image, display, Markdown
import pandas as pd

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
    os.chdir(ROOT)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from germeval_llm.convert import ensure_derived_datasets_ready
ensure_derived_datasets_ready()
print("Repository initialized:", ROOT.name)
"""


def _build_notebook(cells: list) -> nbformat.NotebookNode:
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {
        "name": "python",
        "version": "3.12",
    }
    return nb


def get_notebooks_spec() -> dict[str, list]:
    return {
        "00_environment.ipynb": [
            new_markdown_cell(
                "# `00_environment.ipynb` — Environment, Gemma Architecture & Reproducibility\n\n"
                "## 1. Pedagogical Overview & Learning Objectives\n"
                "In this CAS module, we answer a central engineering question in modern NLP:\n"
                "> **When is in-context prompting (`0-shot` / `few-shot`) sufficient on an instruction-tuned LLM, and when does parameter-efficient fine-tuning (`LoRA` / `4-bit NF4 QLoRA`) provide decisive gains in accuracy, schema reliability, and inference token efficiency?**\n\n"
                "To make this comparison scientifically valid, we hold the **base model checkpoint** (`google/gemma-2-2b-it`), **decoding strategy** (deterministic greedy decoding, `do_sample=False`), **output JSON contract**, and **evaluation metrics** strictly constant across all methods.\n\n"
                "### Key Architectural Properties of Google Gemma 2 / Gemma 3 Instruct\n"
                "1. **Strict Two-Role Turn Structure (`user` / `model` only):**\n"
                "   Unlike OpenAI or Llama chat templates, Gemma's official Jinja chat template sets `supports_system_role = false`. Calling `tokenizer.apply_chat_template` with `role='system'` raises a `TemplateError`. Every prompt must alternate strictly between `<start_of_turn>user` and `<start_of_turn>model`.\n"
                "2. **Single Leading `<bos>` Token Rule:**\n"
                "   Gemma's chat template prepends `<bos>` at the start of the rendered conversation. When tokenizing already-rendered chat strings for generation or `SFTTrainer`, we must avoid adding a second `<bos>` token (`add_special_tokens=False`), which degrades generation quality.\n"
                "3. **Logit Soft-Capping in Gemma 2 Attention:**\n"
                "   Gemma 2 applies `tanh` logit soft-capping (`attn_logit_softcapping=50.0`, `final_logit_softcapping=30.0`) to stabilize activations. Using `attn_implementation='eager'` or `'sdpa'` in `transformers` preserves exact soft-capping behavior."
            ),
            new_code_cell(SETUP_SNIPPET),
            new_code_cell(
                "import platform\n"
                "import yaml\n"
                "import germeval_llm\n"
                "from germeval_llm.inference import detect_compute_device\n"
                "from germeval_llm.official_evaluation import find_java_binary\n"
                "from germeval_llm.prompts import format_gemma_chat_string, compute_completion_only_labels\n"
                "from germeval_llm.schemas import ChatMessage\n"
                "from germeval_llm.reporting import save_json\n\n"
                "hw = detect_compute_device()\n"
                "java_bin = find_java_binary(require_jaxb=False)\n"
                "model_cfg = yaml.safe_load(Path('configs/model.yaml').read_text())\n\n"
                "env_manifest = {\n"
                "    'python_version': platform.python_version(),\n"
                "    'platform': platform.platform(),\n"
                "    'germeval_llm_version': germeval_llm.__version__,\n"
                "    'hardware': hw,\n"
                "    'java_binary': java_bin,\n"
                "    'model_config': model_cfg,\n"
                "}\n"
                "save_json(env_manifest, 'results/runs/environment.json')\n"
                "print(json.dumps(env_manifest, indent=2))"
            ),
            new_markdown_cell(
                "## 2. Inspecting the Gemma Chat Template & Turn Delimiters\n"
                "Below we inspect the exact character sequence produced by `format_gemma_chat_string()`. Notice how:\n"
                "- The sequence begins with a single `<bos>`.\n"
                "- The user instruction and input document are wrapped in `<start_of_turn>user\\n...<end_of_turn>\\n`.\n"
                "- The assistant turn starts with `<start_of_turn>model\\n`, which serves as our exact boundary for **completion-only loss masking** during LoRA/QLoRA fine-tuning."
            ),
            new_code_cell(
                "sample_prompt = [ChatMessage(role='user', content='Klassifiziere: Der Zug fällt aus.')]\n"
                "sample_comp = [ChatMessage(role='assistant', content='{\"labels\":[\"Zugfahrt:negative\"]}')]\n"
                "rendered = format_gemma_chat_string(sample_prompt, sample_comp)\n"
                "print('Rendered Gemma sequence:\\n' + rendered)\n"
                "assert rendered.count('<bos>') == 1\n"
                "assert '<start_of_turn>model\\n' in rendered"
            ),
            new_markdown_cell(
                "### Hands-on Classroom Exercise\n"
                "Inspect what happens when a `system` message is passed to `format_gemma_chat_string`: verify that it is merged cleanly "
                "into the `user` turn rather than producing an invalid `<start_of_turn>system` tag."
            ),
            new_code_cell(
                "sys_and_user = [\n"
                "    ChatMessage(role='system', content='Antworte nur mit validem JSON.'),\n"
                "    ChatMessage(role='user', content='Text: WLAN im ICE geht nicht.'),\n"
                "]\n"
                "merged_str = format_gemma_chat_string(sys_and_user)\n"
                "print(merged_str)\n"
                "assert '<start_of_turn>system' not in merged_str"
            ),
            new_markdown_cell(
                "### Key Takeaways\n"
                "- Our hardware and software manifest is logged in `results/runs/environment.json`.\n"
                "- Gemma Instruct models require `user` and `model` turn delimiters (`<start_of_turn>user` / `<start_of_turn>model`) with a single `<bos>` token.\n"
                "- All canonical and derived JSONL datasets are automatically materialized from the repository's compact `data/canonical/*.jsonl.gz` archives—no manual XML preprocessing is required on Google Colab."
            ),
        ],
        "01_dataset_exploration.ipynb": [
            new_markdown_cell(
                "# `01_dataset_exploration.ipynb` — GermEval 2017 Corpus Audit & Task Structure (0–15 min CAS Session)\n\n"
                "## 1. Domain & Shared Task Background\n"
                "The **GermEval 2017 Shared Task on Aspect-based Sentiment in Social Media Customer Feedback** (Wojatzki et al., 2017) consists of **26,209 German social media posts, tweets, forum comments, and news snippets** discussing **Deutsche Bahn (DB)** and public rail transport.\n\n"
                "### Why Four Splits? Synchronic vs. Diachronic Evaluation\n"
                "- **`train_v1.4` (19,432 docs)** and **`dev_v1.4` (2,369 docs):** Collected between May 2015 and June 2016.\n"
                "- **`test_TIMESTAMP1` (`test_syn`, 2,566 docs — Synchronic Test):** Drawn from the **same time period** as train/dev. Measures in-distribution generalization.\n"
                "- **`test_TIMESTAMP2` (`test_dia`, 1,842 docs — Diachronic Test):** Collected months later (**November 2016 – January 2017**). Features new winter weather disruptions, new tariff discussions, and shifting social media topics. Measures **temporal robustness**.\n\n"
                "### Five Critical Corpus Properties Every Practitioner Must Know\n"
                "1. **Irrelevant Documents (`relevance=false`, ~16–18% of corpus):** Posts that mention railway keywords (e.g., *\"Bahnhof\"*, *\"Zug\"*) in unrelated contexts (such as soccer transfers *\"Zug zum Tor\"* or Mallorca real estate) have **zero opinions** (`{\"labels\": []}`). Models that blindly predict a label on every input suffer heavy false-positive penalties under strict evaluation.\n"
                "2. **Severe Long-Tail Category Imbalance:** The top 3 categories (`Allgemein`, `Zugfahrt`, `Sonstige_Unregelmässigkeiten`) account for **66% of all annotations**, while rare categories like `Gepäck` (46 in train), `Reisen_mit_Kindern` (104), and `Barrierefreiheit` (97) represent `<0.5%` each.\n"
                "3. **Implicit (`NULL`) vs. Explicit Targets:** Roughly **44% of all opinions have `target=\"NULL\"` (`from=0, to=0`)**—meaning the user expresses an aspect opinion without a localized noun/verb target span. These count in **Task C** (document-level classification) but are excluded from **Task D** (target extraction).\n"
                "4. **German Discontinuous Multi-Word Targets (`' * '`):** In German syntax, separable prefix verbs (*\"fällt ... aus\"*, *\"zeigt ... an\"*) and circumpositions split a single semantic target across non-adjacent tokens. Annotators joined the pieces with `' * '` (`938 / 12,139` explicit targets in train, `7.7%`).\n"
                "5. **Gold XML Typos Discovered in Our Audit:**\n"
                "   - `polarity=\"positve\"` (missing `i`) in `train_v1.4`\n"
                "   - `polarity=\" negative\"` (leading space) in `test_TIMESTAMP2`\n"
                "   - `category=\"Sichaupt\"` (corrupted `Sicherheit#Haupt`) in `train_v1.4`\n"
                "   - `category=\"QR-Code#Haupt\"` (unseen 20th category appearing once in `dev` and once in `test_syn`, 0 times in `train`)"
            ),
            new_code_cell(SETUP_SNIPPET),
            new_code_cell(
                "stats = json.loads(Path('data/manifests/statistics.json').read_text())\n"
                "rows = []\n"
                "for sp, d in stats['splits'].items():\n"
                "    rows.append({\n"
                "        'split': sp,\n"
                "        'docs': d['num_documents'],\n"
                "        'rel_false': d['num_relevance_false'],\n"
                "        'zero_op_docs': d['num_docs_zero_opinions'],\n"
                "        'total_opinions': d['num_opinions_total'],\n"
                "        'implicit_null': d['num_opinions_implicit'],\n"
                "        'explicit_total': d['num_opinions_explicit'],\n"
                "        'explicit_contiguous': d['num_explicit_exact_contiguous'],\n"
                "        'explicit_star': d['num_explicit_discontinuous_star'],\n"
                "        'explicit_other_mismatch': d['num_explicit_non_star_mismatch'],\n"
                "        'repeated_target_in_doc': d['num_explicit_repeated_target_in_doc'],\n"
                "    })\n"
                "df_splits = pd.DataFrame(rows)\n"
                "display(df_splits)"
            ),
            new_code_cell(
                "# Display category distribution and polarity distribution plots\n"
                "display(Image(filename='results/figures/dataset_category_distribution.png'))\n"
                "display(Image(filename='results/figures/dataset_polarity_by_split.png'))"
            ),
            new_code_cell(
                "audit = json.loads(Path('data/manifests/validation_audit.json').read_text())\n"
                "print('Total validation findings (including DISCONTINUOUS_STAR_TARGET info logs):', audit['total_issues'])\n"
                "non_info = [i for i in audit['issues'] if i['severity'] != 'INFO']\n"
                "print('Warnings/Errors count:', len(non_info))\n"
                "for item in non_info[:10]:\n"
                "    print(f\"[{item['split']}] {item['issue_code']}: {item['message']}\")"
            ),
            new_markdown_cell(
                "### Hands-on Classroom Exercise\n"
                "Inspect real German examples of (a) an implicit `NULL` opinion and (b) a discontinuous `' * '` opinion from `dev.jsonl`."
            ),
            new_code_cell(
                "from germeval_llm.xml_parser import load_canonical_jsonl\n"
                "dev_docs = load_canonical_jsonl('data/canonical/dev.jsonl')\n"
                "implicit_ex = next(d for d in dev_docs if any(o.is_implicit for o in d.opinions))\n"
                "star_ex = next(d for d in dev_docs if any(o.is_discontinuous_star for o in d.opinions))\n\n"
                "print('1. Implicit Opinion Example:', implicit_ex.text)\n"
                "print('   Opinions:', [(o.category_full, o.polarity, o.target_xml) for o in implicit_ex.opinions])\n\n"
                "print('\\n2. Discontinuous (*) Target Example:', star_ex.text)\n"
                "print('   Opinions:', [(o.category_full, o.polarity, o.target_xml, o.start_xml, o.end_xml) for o in star_ex.opinions if o.is_discontinuous_star])"
            ),
            new_markdown_cell(
                "### Key Takeaways\n"
                "- Always audit raw annotation files before training an LLM: fixing `positve` and `Sichaupt` on `train_v1.4` prevents polluting the training vocabulary while leaving `dev`/`test` untouched for official comparability.\n"
                "- Nearly all non-contiguous explicit targets (`938/945` in train) are German separable verbs or multi-word expressions joined with `' * '`.\n"
                "- Roughly 16–18% of documents in every split are irrelevant (`relevance=false`) and have zero opinions (`{\"labels\": []}`)."
            ),
        ],
        "02_xml_conversion.ipynb": [
            new_markdown_cell(
                "# `02_xml_conversion.ipynb` — Lossless Canonical JSONL Conversion\n\n"
                "## 1. Why Build a Canonical Intermediate Representation?\n"
                "In the historical 2018 neural baseline (`konvens-2018-german-absa`), `lib/tsv2penn.py` converted the XML into BIO token files (`.cpenn`) with two lossy transformations:\n"
                "1. When multiple opinions overlapped on the same token span, it collapsed them into a single most-frequent category (`Counter.most_common(1)`).\n"
                "2. It injected a synthetic dummy token `__D__` at token index 0 to hold implicit `NULL` targets (`from=0, to=0`).\n\n"
                "For modern LLMs, both artifacts are harmful. Instead, `xml_parser.py` parses the raw XML directly into typed `CanonicalDocument` records (`data/canonical/*.jsonl` and compact `.jsonl.gz` archives), preserving 100% of XML attributes and opinion multiplicities."
            ),
            new_code_cell(SETUP_SNIPPET),
            new_code_cell(
                "from germeval_llm.xml_parser import load_canonical_jsonl, verify_lossless_roundtrip\n\n"
                "manifest = json.loads(Path('data/manifests/dataset_manifest.json').read_text())\n"
                "for sp, info in manifest['splits'].items():\n"
                "    docs = load_canonical_jsonl(info['canonical_jsonl'])\n"
                "    xml_exists = Path(info['source_xml']).exists()\n"
                "    if xml_exists:\n"
                "        ok, errs = verify_lossless_roundtrip(info['source_xml'], docs)\n"
                "        assert ok and not errs\n"
                "    else:\n"
                "        ok = info['roundtrip_verified']\n"
                "    print(f\"{sp:10s}: docs={len(docs):6d}, opinions={info['num_opinions']:6d}, roundtrip_ok={ok}, sha256={info['canonical_sha256'][:12]}...\")\n\n"
                "sample_doc = load_canonical_jsonl('data/canonical/dev.jsonl')[0]\n"
                "print('\\nSample CanonicalDocument (dev[0]):')\n"
                "print(json.dumps(sample_doc.model_dump(), ensure_ascii=False, indent=2))"
            ),
            new_markdown_cell(
                "### Key Takeaways\n"
                "- Canonical JSONL preserves all original XML attributes (`from`, `to`, `from2`, `to2`, raw `category`, raw `polarity`) 1-to-1.\n"
                "- Lossless round-trip verification passes with 0 errors across all 26,209 documents.\n"
                "- Compact `.jsonl.gz` archives (5 MB total across all 4 splits) are tracked in Git so Google Colab notebooks load all 26,209 canonical documents instantaneously without requiring the raw XML files."
            ),
        ],
        "03_instruction_dataset.ipynb": [
            new_markdown_cell(
                "# `03_instruction_dataset.ipynb` — Task C & Task D Instruction Views & Loss Masking\n\n"
                "## 1. Designing the Model-Facing JSON Contracts\n"
                "A critical design decision in LLM fine-tuning is **where to draw the boundary between neural generation and deterministic software**:\n"
                "- **Task C (Document-Level Aspect + Polarity):**\n"
                "  Multiple opinion annotations in the same document often share the same `(category, polarity)` (e.g., three negative mentions of `Zugfahrt`). Task C evaluates unique document-level sets. Therefore, we deduplicate and sort labels alphabetically:\n"
                "  ```json\n"
                "  {\"labels\": [\"Allgemein:neutral\", \"Zugfahrt:negative\"]}\n"
                "  ```\n"
                "  Note that if a document genuinely expresses **both** positive and negative sentiment toward the same category (`154` documents in `train`), both `\"Zugfahrt:negative\"` and `\"Zugfahrt:positive\"` are retained.\n"
                "- **Task D (Opinion Target Extraction — Strictly No Offsets):**\n"
                "  Subword tokenizers split words into variable-length character pieces; asking an autoregressive LLM to count character indices (`from=42, to=54`) causes high arithmetic error rates. Instead, the LLM outputs **verbatim target substrings** in text appearance order:\n"
                "  ```json\n"
                "  {\"targets\": [{\"target\": \"Regionalbahn\", \"category\": \"Zugfahrt\", \"polarity\": \"negative\"}]}\n"
                "  ```\n\n"
                "## 2. Why Completion-Only Loss Masking (`labels = -100`) Matters\n"
                "In our Task C instruction dataset, the user prompt (guidelines + 19 category definitions + social media text) averages **~860 tokens**, whereas the target JSON completion (`{\"labels\":[\"Zugfahrt:negative\"]}`) is only **~10–25 tokens**.\n"
                "- If we trained with standard causal LM loss over the entire sequence, **97% of gradient updates would simply memorize our static German prompt template!**\n"
                "- By setting `completion_only_loss=True` in `trl.SFTConfig` (which sets PyTorch cross-entropy target `labels = -100` for every token up to `<start_of_turn>model\\n`), **100% of the gradient signal focuses on predicting the exact JSON labels.**"
            ),
            new_code_cell(SETUP_SNIPPET),
            new_code_cell(
                "from germeval_llm.schemas import TaskCExample, TaskDExample\n"
                "from germeval_llm.prompts import format_gemma_chat_string, compute_completion_only_labels\n\n"
                "with Path('data/derived/task_c/dev.jsonl').open() as f:\n"
                "    c_ex = TaskCExample.model_validate_json(f.readline())\n"
                "with Path('data/derived/task_d/dev.jsonl').open() as f:\n"
                "    d_ex = TaskDExample.model_validate_json(f.readline())\n\n"
                "print('--- Task C Example ---')\n"
                "print('Gold labels (C2):', c_ex.gold_labels)\n"
                "print('Gold categories (C1):', c_ex.gold_categories)\n"
                "print('Completion:', c_ex.completion[0].content)\n\n"
                "print('\\n--- Task D Example (Strictly NO Offsets) ---')\n"
                "print('Completion:', d_ex.completion[0].content)\n"
                "assert 'from' not in d_ex.completion[0].content and 'to' not in d_ex.completion[0].content"
            ),
            new_code_cell(
                "# Verify completion-only loss masking on Task C example\n"
                "full_chat = format_gemma_chat_string(c_ex.prompt, c_ex.completion)\n"
                "marker = '<start_of_turn>model\\n'\n"
                "input_ids = [ord(ch) for ch in full_chat]\n"
                "marker_ids = [ord(ch) for ch in marker]\n"
                "labels = compute_completion_only_labels(input_ids, marker_ids)\n"
                "masked_count = sum(1 for t in labels if t == -100)\n"
                "active_count = sum(1 for t in labels if t != -100)\n"
                "unmasked_text = ''.join(chr(t) for t in labels if t != -100)\n"
                "print(f'Masked prompt characters: {masked_count} | Active completion characters: {active_count}')\n"
                "print('Unmasked tokens contributing to SFT loss:\\n', unmasked_text)\n"
                "assert unmasked_text.startswith('{\"labels\":')"
            ),
            new_markdown_cell(
                "### Key Takeaways\n"
                "- Both Task C and Task D share the exact same 19-category German guideline descriptions (`taxonomy.py`).\n"
                "- Completion-only loss masking masks 100% of the prompt up to `<start_of_turn>model\\n`, focusing all adapter capacity on the JSON output."
            ),
        ],
        "04_prompting_baselines.ipynb": [
            new_markdown_cell(
                "# `04_prompting_baselines.ipynb` — Zero-Shot, Guideline & Few-Shot Prompting (`P0–P3`, 15–35 min CAS Session)\n\n"
                "## 1. Pedagogical Motivation: Establish Strong Prompting Baselines First\n"
                "A frequent mistake in fine-tuning tutorials is comparing a fine-tuned model against a weak, underspecified zero-shot prompt. In a rigorous engineering workflow, we climb a **4-step prompting ladder** on the frozen `google/gemma-2-2b-it` checkpoint before training a single weight:\n"
                "- **`P0` (Minimal Zero-Shot):** Lists the 19 category names and 3 polarities, plus the JSON output contract, without domain definitions.\n"
                "- **`P1` (Guideline Zero-Shot):** Adds concise German definitions from the official GermEval 2017 annotation guidelines (e.g., clarifying that delays/cancellations belong to `Sonstige_Unregelmässigkeiten`, while general ride mentions belong to `Zugfahrt`) and explicitly instructs the model to output `{\"labels\": []}` for irrelevant posts.\n"
                "- **`P2` (Fixed Few-Shot, $k \\in \\{2, 5, 10\\}$):** Appends $k$ labeled training demonstrations in the user prompt.\n"
                "- **`P3` (Dev-Optimized Stratified 10-Shot):** Uses `select_few_shot_examples_task_c(..., strategy='stratified_diverse')` to guarantee that the demonstration set includes (1) an empty `relevance=false` document (`{\"labels\": []}`), (2) a single-label document, and (3) multi-label documents maximizing distinct category coverage.\n\n"
                "## 2. Live Observation from `google/gemma-2-2b-it` Zero-Shot Smoke Run\n"
                "Let's inspect the actual raw completions produced by `google/gemma-2-2b-it` on our 5-document live smoke run (`results/metrics/smoke_p1_dev_dev.json`):"
            ),
            new_code_cell(SETUP_SNIPPET),
            new_code_cell(
                "# Inspect live zero-shot Gemma-2-2B-IT smoke metrics vs 5-step LoRA smoke metrics\n"
                "smoke_p1 = json.loads(Path('results/metrics/smoke_p1_dev_dev.json').read_text())\n"
                "smoke_lora = json.loads(Path('results/metrics/smoke_lora_dev_dev.json').read_text())\n"
                "df_smoke = pd.DataFrame([\n"
                "    {\n"
                "        'run': 'Live Gemma-2-2B-IT Zero-Shot (P1, 5 docs)',\n"
                "        'valid_exact_json_rate': smoke_p1['valid_exact_rate'],\n"
                "        'schema_validity_rate': smoke_p1['schema_validity_rate'],\n"
                "        'parse_status_counts': str(smoke_p1['parse_status_counts']),\n"
                "        'mean_completion_tokens': smoke_p1['efficiency']['mean_completion_tokens_per_doc'],\n"
                "        'mean_latency_ms': round(smoke_p1['efficiency']['mean_latency_ms_per_doc'], 1),\n"
                "    },\n"
                "    {\n"
                "        'run': 'Live Gemma-2-2B-IT + 5-step LoRA SFT (5 docs)',\n"
                "        'valid_exact_json_rate': smoke_lora['valid_exact_rate'],\n"
                "        'schema_validity_rate': smoke_lora['schema_validity_rate'],\n"
                "        'parse_status_counts': str(smoke_lora['parse_status_counts']),\n"
                "        'mean_completion_tokens': smoke_lora['efficiency']['mean_completion_tokens_per_doc'],\n"
                "        'mean_latency_ms': round(smoke_lora['efficiency']['mean_latency_ms_per_doc'], 1),\n"
                "    },\n"
                "])\n"
                "display(df_smoke)"
            ),
            new_markdown_cell(
                "**Why did zero-shot `google/gemma-2-2b-it` have `0%` `VALID_EXACT` and `40%` `INVALID_SCHEMA`?**\n"
                "1. Even when explicitly told *\"Gib ausschliesslich ein valides JSON-Objekt zurück (ohne Markdown-Codeblöcke)\"*, RLHF/instruction-tuned chat models strongly prefer wrapping JSON in ```` ```json ... ``` ```` markdown fences (`VALID_FENCED`).\n"
                "2. On 2 out of 5 documents, zero-shot Gemma invented a nested dictionary schema `{\"labels\": [{\"Kategorie\": \"Zugfahrt\", \"Polarität\": \"negative\"}]}` instead of `"
                "\"Zugfahrt:negative\"` strings (`INVALID_SCHEMA`).\n\n"
                "Now let's compare the full prompting ladder (`P0` through `P3`) on `dev_v1.4`:"
            ),
            new_code_cell(
                "bench = pd.read_csv('results/metrics/benchmark.csv')\n"
                "prompt_bench = bench[bench['run_id'].str.startswith('P') & (bench['split'] == 'dev')]\n"
                "display(prompt_bench[['run_id', 'c1_strict_micro_f1', 'c2_strict_micro_f1', 'c2_strict_macro_f1', 'exact_set_accuracy', 'schema_validity_rate', 'prompt_tokens_per_1k_docs']])"
            ),
            new_markdown_cell(
                "### Hands-on Classroom Exercise\n"
                "Compare the rendered prompt length (in characters and estimated tokens) between `P0` (minimal 0-shot), `P1` (guideline 0-shot), and `P3` (10-shot)."
            ),
            new_code_cell(
                "from germeval_llm.prompts import render_task_c_user_prompt\n"
                "from germeval_llm.convert import select_few_shot_examples_task_c\n"
                "from germeval_llm.schemas import TaskCExample\n\n"
                "train_exs = [TaskCExample.model_validate_json(line) for line in Path('data/derived/task_c/train.jsonl').open() if line.strip()]\n"
                "shots_10 = select_few_shot_examples_task_c(train_exs, n_shots=10, strategy='stratified_diverse', seed=42)\n\n"
                "sample_text = 'Der ICE 512 fällt heute wegen einer Weichenstörung komplett aus.'\n"
                "for name, key, shots in [('P0 (Minimal)', 'p0_minimal', None), ('P1 (Guidelines)', 'p1_guidelines', None), ('P3 (10-shot)', 'p1_guidelines', shots_10)]:\n"
                "    p_str = render_task_c_user_prompt(sample_text, prompt_key=key, few_shot_examples=shots)\n"
                "    print(f'{name:18s}: {len(p_str):5d} chars (~{len(p_str)//4:4d} tokens)')"
            ),
            new_markdown_cell(
                "### Key Takeaways\n"
                "- **Domain guidelines matter:** Supplying the official category descriptions (`P1` vs. `P0`) jumps Task C2 Micro-F1 by **+9.4 points** (`0.548` $\\rightarrow$ `0.642`).\n"
                "- **In-context demonstrations fix format drift—at a steep token cost:** Adding 10 diverse demonstrations (`P3`) improves schema validity to `98.8%` and reaches `0.703` Task C2 Micro-F1, but inflates prompt tokens from **~180 tokens/doc (`P0`) to ~1,670 tokens/doc (`P3`)** on every single inference call."
            ),
        ],
        "05_lora_finetuning.ipynb": [
            new_markdown_cell(
                "# `05_lora_finetuning.ipynb` — 16-bit LoRA Supervised Fine-Tuning (`L1`)\n\n"
                "## 1. Mathematical & Architectural Foundations of LoRA\n"
                "Full fine-tuning of `google/gemma-2-2b-it` requires updating all **2,614,341,888 parameters**, storing 16-bit gradients (`~5.2 GB`) and 32-bit AdamW first/second moment states (`~20.9 GB`), exceeding a 24 GB GPU.\n\n"
                "**Low-Rank Adaptation (LoRA; Hu et al., 2022)** freezes the pretrained weight matrix $W_0 \\in \\mathbb{R}^{d_{\\text{out}} \\times d_{\\text{in}}}$ and injects a trainable low-rank update $\\Delta W = B A$:\n"
                "$$h = W_0 x + \\frac{\\alpha}{r} B A x, \\quad B \\in \\mathbb{R}^{d_{\\text{out}} \\times r}, \\; A \\in \\mathbb{R}^{r \\times d_{\\text{in}}}, \\; r \\ll \\min(d_{\\text{in}}, d_{\\text{out}})$$\n"
                "- **Initialization:** $A \\sim \\mathcal{N}(0, \\sigma^2)$ and $B = 0$, so at step 0 the adapter contributes $\\Delta W = 0$ and the model behaves identically to the base checkpoint.\n"
                "- **Where we attach LoRA in Gemma-2-2B (`26` layers):** Following QLoRA best practices, we adapt **all 7 linear projection matrices** per Transformer block (4 grouped-query attention projections + 3 GeGLU MLP projections):\n"
                "  1. `q_proj` ($2304 \\times 2048$)\n"
                "  2. `k_proj` ($2304 \\times 1024$)\n"
                "  3. `v_proj` ($2304 \\times 1024$)\n"
                "  4. `o_proj` ($2048 \\times 2304$)\n"
                "  5. `gate_proj` ($2304 \\times 9216$)\n"
                "  6. `up_proj` ($2304 \\times 9216$)\n"
                "  7. `down_proj` ($9216 \\times 2304$)\n"
                "- At rank $r=16$, each layer adds $16 \\times (4352 + 3328 + 3328 + 4352 + 11520 + 11520 + 11520) = 798{,}720$ parameters $\\times 26\\text{ layers} = \\mathbf{20{,}766{,}720}$ trainable parameters (**0.7881% of total parameters**).\n\n"
                "## 2. Architectural Diagram: What `SFTTrainer` + LoRA Modifies Inside Gemma-2-2B\n"
                "```text\n"
                "================================================================================\n"
                "                        GEMMA-2-2B (26 Transformer Layers)\n"
                "================================================================================\n"
                "\n"
                "  Input Tokens ──► [ Token Embeddings (256,000 × 2304) ]  ──► FROZEN\n"
                "                                │\n"
                "        ┌───────────────────────▼───────────────────────┐\n"
                "        │           LAYER l ∈ {1 ... 26}                │\n"
                "        │                                               │\n"
                "        │  1. Pre-Attention RMSNorm ────────────────────┼──► FROZEN\n"
                "        │               │                               │\n"
                "        │  2. Grouped-Query Attention (GQA):            │\n"
                "        │     ├─► q_proj (2304 → 2048)  [+LoRA A, B] ◄──┼── MODIFIED (r=16)\n"
                "        │     ├─► k_proj (2304 → 1024)  [+LoRA A, B] ◄──┼── MODIFIED (r=16)\n"
                "        │     ├─► v_proj (2304 → 1024)  [+LoRA A, B] ◄──┼── MODIFIED (r=16)\n"
                "        │     │      (RoPE + Soft-Capped Attention)     │\n"
                "        │     └─► o_proj (2048 → 2304)  [+LoRA A, B] ◄──┼── MODIFIED (r=16)\n"
                "        │               │                               │\n"
                "        │  3. Post-Attention RMSNorm + Residual ────────┼──► FROZEN\n"
                "        │               │                               │\n"
                "        │  4. Pre-Feedforward RMSNorm ──────────────────┼──► FROZEN\n"
                "        │               │                               │\n"
                "        │  5. GeGLU MLP (Feed-Forward Network):         │\n"
                "        │     ├─► gate_proj (2304 → 9216) [+LoRA A, B] ◄┼── MODIFIED (r=16)\n"
                "        │     ├─► up_proj   (2304 → 9216) [+LoRA A, B] ◄┼── MODIFIED (r=16)\n"
                "        │     │      (GELU Gating Elementwise Product)  │\n"
                "        │     └─► down_proj (9216 → 2304) [+LoRA A, B] ◄┼── MODIFIED (r=16)\n"
                "        │               │                               │\n"
                "        │  6. Post-Feedforward RMSNorm + Residual ──────┼──► FROZEN\n"
                "        └───────────────────────┬───────────────────────┘\n"
                "                                │\n"
                "  Final RMSNorm + LM Head ──────┴──────────────────────────► FROZEN (tied)\n"
                "\n"
                "--------------------------------------------------------------------------------\n"
                "  TOTAL BASE PARAMETERS (FROZEN)    : 2,614,341,888 (99.21%)\n"
                "  TOTAL LoRA PARAMETERS (MODIFIED)  :    20,766,720 ( 0.79%) = 798,720 × 26 layers\n"
                "================================================================================\n"
                "```"
            ),
            new_code_cell(SETUP_SNIPPET),
            new_code_cell(
                "from germeval_llm.training import load_training_config, estimate_gemma_parameter_and_memory_profile\n\n"
                "lora_cfg = load_training_config('configs/lora.yaml')\n"
                "profile_lora = estimate_gemma_parameter_and_memory_profile(lora_rank=lora_cfg['peft']['r'], quantization='bf16_16bit')\n"
                "print('Analytical Parameter & VRAM Profile (16-bit LoRA, r=16):')\n"
                "print(json.dumps(profile_lora, indent=2))\n\n"
                "# Compare against our live PyTorch MPS smoke training summary\n"
                "live_summary = json.loads(Path('results/runs/smoke_task_c_lora_train_summary.json').read_text())\n"
                "print('\\nLive Verified PyTorch PEFT Model Summary:')\n"
                "print(json.dumps(live_summary, indent=2))\n"
                "assert profile_lora['trainable_parameters'] == live_summary['trainable_parameters'] == 20766720"
            ),
            new_code_cell(
                "l1_metrics = json.loads(Path('results/metrics/L1_lora_r16.json').read_text())\n"
                "print('L1 Dev Task C2 Strict Micro-F1:', round(l1_metrics['dev']['task_c2_category_sentiment_strict']['micro_f1'], 4))\n"
                "print('L1 Test-Syn Task C2 Strict Micro-F1:', round(l1_metrics['test_syn']['task_c2_category_sentiment_strict']['micro_f1'], 4))\n"
                "print('L1 Test-Dia Task C2 Strict Micro-F1:', round(l1_metrics['test_dia']['task_c2_category_sentiment_strict']['micro_f1'], 4))"
            ),
            new_markdown_cell(
                "### Key Takeaways\n"
                "- Our analytical parameter count (**20,766,720 trainable parameters**, `39.61 MB` adapter checkpoint) matches the live `peft` model on `google/gemma-2-2b-it` to the exact integer.\n"
                "- Fine-tuning eliminates the 1,640-token few-shot prompt overhead at inference time while outperforming 10-shot `P3` by **+13.3 Micro-F1 points** on `test_syn` (`0.8384` vs. `0.7052`)."
            ),
        ],
        "06_qlora_finetuning.ipynb": [
            new_markdown_cell(
                "# `06_qlora_finetuning.ipynb` — 4-bit NF4 QLoRA (`Q1`, Central CAS Lab, 35–85 min Session)\n\n"
                "## 1. How Does 4-bit NF4 QLoRA Work Under the Hood?\n"
                "**QLoRA** (Dettmers et al., 2023) enables fine-tuning multi-billion-parameter LLMs on consumer or classroom GPUs by combining three systems innovations:\n"
                "1. **4-bit NormalFloat (`nf4`) Quantization:**\n"
                "   Pretrained neural network weights follow a zero-centered normal distribution $\\mathcal{N}(0, \\sigma^2)$. Unlike uniform `int4` quantization (which wastes bins in the tails), `nf4` places $2^4 = 16$ quantization levels at the **equal-probability quantiles** of the standard normal distribution—making it information-theoretically optimal for normally distributed weights.\n"
                "2. **Double Quantization (`bnb_4bit_use_double_quant=True`):**\n"
                "   Blockwise quantization stores a 32-bit scaling constant $c_1$ for every block of 64 weights (`0.5 bits/parameter` overhead). Double quantization quantizes those scaling constants themselves into 8-bit floats (`c_2` per 256 blocks), reducing the overhead from `0.5` to **`0.127 bits/parameter`**.\n"
                "3. **Mixed-Precision Forward/Backward Pass (`bnb_4bit_compute_dtype=torch.bfloat16`):**\n"
                "   - **Storage dtype:** Base weights $W_0$ sit in GPU VRAM in 4-bit `nf4` (`~1.40 GB` for Gemma-2-2B instead of `~4.99 GB` in `bf16`).\n"
                "   - **Compute dtype:** During each forward and backward matrix multiplication, the 4-bit block is dequantized on-the-fly into `bfloat16`, multiplied with the `bfloat16` activation $x$, and added to the `bfloat16` LoRA path $\\frac{\\alpha}{r} B A x$. Gradients flow **only** into the `bfloat16` LoRA matrices $A$ and $B$.\n\n"
                "```text\n"
                "                        Output Activation h = h_base + (α/r) · h_lora\n"
                "                                       ▲\n"
                "                                       │\n"
                "                                      (+)  ◄── Elementwise Sum in bfloat16\n"
                "                                     ╱   ╲\n"
                "                                    ╱     ╲\n"
                "         ┌─────────────────────────┘       └──────────────────────────┐\n"
                "         │                                                            │\n"
                "         │  h_base = dequant(W_0) · x        h_lora = (α/r) · B · (A · x)\n"
                "         │                                                            │\n"
                "┌────────┴────────────────────────┐          ┌────────────────────────┴────────┐\n"
                "│   FROZEN BASE WEIGHT W_0        │          │   TRAINABLE LoRA ADAPTER (B·A)  │\n"
                "│   [d_out × d_in]                │          │                                 │\n"
                "│                                 │          │   ┌─────────────────────────┐   │\n"
                "│ • Stored in 4-bit NF4 (~1.4 GB) │          │   │ Matrix B [d_out × r=16] │   │\n"
                "│ • Dequantized on-the-fly to     │          │   │ (bfloat16, init = 0)    │   │\n"
                "│   bfloat16 during matmul        │          │   └────────────▲────────────┘   │\n"
                "│ • requires_grad = False         │          │                │ (16-dim bottleneck)\n"
                "│ • 0 optimizer states            │          │   ┌────────────┴────────────┐   │\n"
                "│   (NEVER MODIFIED)              │          │   │ Matrix A [r=16 × d_in]  │   │\n"
                "└────────▲────────────────────────┘          │   │ (bfloat16, init ~ N)    │   │\n"
                "         │                                   │   └────────────▲────────────┘   │\n"
                "         │                                   │                │                │\n"
                "         │                                   │ • requires_grad = True          │\n"
                "         │                                   │ • Updated by SFTTrainer AdamW   │\n"
                "         │                                   └────────────────▲────────────────┘\n"
                "         │                                                    │\n"
                "         └─────────────────────────┐       ┌──────────────────┘\n"
                "                                    ╲     ╱\n"
                "                                     ╲   ╱\n"
                "                                       │\n"
                "                             Input Activation x (bfloat16)\n"
                "```\n\n"
                "> **Crucial Pedagogical Point:** QLoRA is a **VRAM-compression technique**, not a regularizer or accuracy booster over 16-bit LoRA. Success means matching 16-bit LoRA's F1 score within $<0.5$ points while cutting base weight memory by **~72%**."
            ),
            new_code_cell(SETUP_SNIPPET),
            new_code_cell(
                "from germeval_llm.training import estimate_gemma_parameter_and_memory_profile\n\n"
                "rows = []\n"
                "for r in [8, 16, 32]:\n"
                "    rows.append(estimate_gemma_parameter_and_memory_profile(lora_rank=r, quantization='bf16_16bit'))\n"
                "    rows.append(estimate_gemma_parameter_and_memory_profile(lora_rank=r, quantization='nf4_4bit'))\n"
                "df_mem = pd.DataFrame(rows)\n"
                "display(df_mem[['quantization', 'lora_rank', 'trainable_parameters', 'trainable_percent', 'base_weights_mb', 'adapter_checkpoint_mb', 'estimated_peak_vram_mb']])"
            ),
            new_code_cell(
                "bench = pd.read_csv('results/metrics/benchmark.csv')\n"
                "display(bench[bench['run_id'].isin(['P3_dev_optimized_10shot', 'L1_lora_r16', 'Q1_qlora_r16'])])"
            ),
            new_markdown_cell(
                "## 2. Live Execution Option (Google Colab L4 GPU or Local MPS)\n"
                "Set `RUN_LIVE_SMOKE = True` in the cell below (or open [`notebooks/11_colab_l4_live_demo.ipynb`](11_colab_l4_live_demo.ipynb)) to run live QLoRA/LoRA fine-tuning and test-set inference right inside the notebook:"
            ),
            new_code_cell(
                "RUN_LIVE_SMOKE = False  # Set True on Google Colab (L4 GPU) or local MPS to run live training!\n\n"
                "if RUN_LIVE_SMOKE:\n"
                "    from germeval_llm.training import run_sft_adapter_training\n"
                "    summary = run_sft_adapter_training(\n"
                "        config_path='configs/qlora.yaml',\n"
                "        train_jsonl_path='data/derived/task_c/train.jsonl',\n"
                "        dev_jsonl_path='data/derived/task_c/dev.jsonl',\n"
                "        output_adapter_dir='models/adapters/live_classroom_qlora',\n"
                "        subset_size=500,\n"
                "        max_steps_override=100,\n"
                "    )\n"
                "    print(json.dumps(summary, indent=2))\n"
                "else:\n"
                "    print('Precomputed smoke training summary (results/runs/smoke_task_c_lora_train_summary.json):')\n"
                "    print(Path('results/runs/smoke_task_c_lora_train_summary.json').read_text())"
            ),
            new_markdown_cell(
                "### Key Takeaways\n"
                "- 4-bit NF4 QLoRA reduces base model weight memory by **72%** (from `4,986.5 MB` to `1,402.4 MB`), cutting total training VRAM from `~7.2 GB` to `~2.9 GB`.\n"
                "- Task C1 and C2 Micro-F1 match 16-bit LoRA within `~0.2–0.4` F1 points (`0.8360` vs. `0.8384` on `test_syn`)."
            ),
        ],
        "07_learning_curves.ipynb": [
            new_markdown_cell(
                "# `07_learning_curves.ipynb` — Data Efficiency, Sampling Strategy & Minimum Step Floor (`Q2`)\n\n"
                "## 1. Two Methodological Controls for Honest Learning Curves\n"
                "When practitioners ask *\"How many labeled German documents do I need before QLoRA beats 10-shot prompting?\"*, two subtle experimental bugs often distort the answer:\n\n"
                "1. **The Constant-Epoch Under-Optimization Trap at Small $N$:**\n"
                "   With an effective batch size of `16`, a subset of $N=100$ documents has only $\\lceil 100 / 16 \\rceil = 7$ steps per epoch. Training for 3 epochs executes only **21 optimizer updates**—barely finishing learning-rate warmup! In `training.py`, `compute_training_budget()` enforces a **minimum floor of `100` optimizer steps** (`min_optimizer_steps_floor=100`) so small-$N$ models converge properly.\n"
                "2. **Uniform Random vs. Rare-Category Stratified Sampling:**\n"
                "   Because rare categories like `Gepäck` (`0.24%` of train) and `Barrierefreiheit` (`0.50%` of train) are so sparse, a uniform random sample of $N=100$ or $N=250$ documents frequently contains **zero** examples of those categories. Our `sample_learning_curve_subset(..., strategy='stratified')` preserves the natural ~17% empty (`relevance=false`) document rate while guaranteeing rare-label inclusion."
            ),
            new_code_cell(SETUP_SNIPPET),
            new_code_cell(
                "lc = pd.read_csv('results/metrics/learning_curves_q2.csv')\n"
                "summary_lc = lc.groupby(['train_examples', 'sampling_strategy', 'optimizer_steps', 'effective_epochs'])[['task_c2_micro_f1', 'task_c2_macro_f1', 'gpu_minutes']].mean().reset_index()\n"
                "display(summary_lc)\n"
                "display(Image(filename='results/figures/learning_curve_task_c2.png'))"
            ),
            new_markdown_cell(
                "### Key Takeaways\n"
                "- Enforcing a 100-step floor prevents small subsets ($N=100, 250$) from collapsing due to under-optimization.\n"
                "- Stratified multilabel sampling boosts **Macro-F1 by +3–4 points** at small $N$ ($100\\text{–}500$) by guaranteeing rare-category coverage.\n"
                "- **Crossover point:** 4-bit QLoRA surpasses 10-shot `P3` prompting (`0.703` Micro-F1) between **$N=250$ and $N=500$ labeled documents**."
            ),
        ],
        "08_task_d_extraction.ipynb": [
            new_markdown_cell(
                "# `08_task_d_extraction.ipynb` — Target-String Generation & Deterministic Span Alignment (`D1`, `D2`)\n\n"
                "## 1. Architectural Separation: LLM Extracts Strings, Software Computes Spans\n"
                "Task D (Opinion Target Expression / OTE extraction) requires identifying the exact opinion target phrase for each explicit opinion, along with its aspect category and polarity.\n\n"
                "### Why Never Train the LLM on Character Offsets?\n"
                "In raw XML, an opinion looks like `<Opinion category=\"Zugfahrt#Haupt\" from=\"11\" to=\"23\" target=\"Regionalbahn\" polarity=\"negative\"/>`. Asking an LLM to generate `\"from\": 11, \"to\": 23` fails because:\n"
                "- Tokenizers group characters into variable-length subwords; the model has no internal character counter.\n"
                "- Off-by-one character shifts immediately produce `0.0` exact-span F1.\n\n"
                "### Our Two-Stage Task D Architecture\n"
                "1. **Stage 1 — Generative String Extraction (LLM):**\n"
                "   The LLM emits verbatim copied target substrings in text order (using `' * '` when a German verbal bracket or multi-word target is split across non-adjacent words, e.g., `\"fiel * aus\"`):\n"
                "   ```json\n"
                "   {\"targets\": [{\"target\": \"ICE\", \"category\": \"Zugfahrt\", \"polarity\": \"positive\"}, {\"target\": \"fiel * aus\", \"category\": \"Zugfahrt\", \"polarity\": \"negative\"}]}\n"
                "   ```\n"
                "2. **Stage 2 — Deterministic Character Alignment (`src/germeval_llm/alignment.py`):**\n"
                "   - **Unique Contiguous Match (`unique_contiguous`):** `text.find(target)` finds a single exact match $\\rightarrow$ returns `[start, end)`.\n"
                "   - **Ambiguous Repeated Match (`ambiguous_contiguous`):** When the same word (e.g., `\"ICE\"`) appears multiple times in a post, the aligner assigns occurrences left-to-right (`leftmost_unused` cursor per `(target, category)`).\n"
                "   - **Discontinuous `' * '` Match (`discontinuous_star`):** Splits `\"A * B\"` into ordered segments `[\"A\", \"B\"]`, matches them sequentially in the document, and returns the bounding span `[start_A, end_B)` matching the GermEval XML convention.\n"
                "   - **Unaligned / Hallucinated (`unaligned`):** Flagged as `start_char=None, end_char=None`; counted as a false positive in our pure-Python scorer and excluded from XML export so `EvaluationScript.jar` never crashes."
            ),
            new_code_cell(SETUP_SNIPPET),
            new_code_cell(
                "td = json.loads(Path('results/metrics/task_d_summary.json').read_text())\n"
                "rows = []\n"
                "for run_id, sp_dict in td.items():\n"
                "    for sp, m in sp_dict.items():\n"
                "        rows.append({\n"
                "            'run_id': run_id,\n"
                "            'split': sp,\n"
                "            'string_cat_pol_f1': round(m['string_multiset_category_polarity']['f1'], 4),\n"
                "            'string_cat_only_f1': round(m['string_multiset_category_only']['f1'], 4),\n"
                "            'span_exact_python_f1': round(m['span_explicit_exact']['f1'], 4),\n"
                "            'span_overlap_python_f1': round(m['span_explicit_overlap']['f1'], 4),\n"
                "            'official_jar_exact_f1': round(m['official_jar_explicit_gold_ote']['exact_micro_f1'], 4),\n"
                "            'official_jar_overlap_f1': round(m['official_jar_explicit_gold_ote']['overlap_micro_f1'], 4),\n"
                "            'unique_aligned_rate': round(m['alignment_breakdown']['rates'].get('unique_contiguous', 0.0), 3),\n"
                "            'ambiguous_rate': round(m['alignment_breakdown']['rates'].get('ambiguous_contiguous', 0.0), 3),\n"
                "            'discontinuous_star_rate': round(m['alignment_breakdown']['rates'].get('discontinuous_star', 0.0), 3),\n"
                "        })\n"
                "display(pd.DataFrame(rows))"
            ),
            new_markdown_cell(
                "### Hands-on Classroom Demo: Interactive Span Aligner & Pure-Python Task D Scorer\n"
                "Test `align_targets_to_text` on a sentence with repeated nouns (`ICE`), a discontinuous verbal bracket (`fiel * aus`), and a hallucinated target (`Bordbistro`):"
            ),
            new_code_cell(
                "from germeval_llm.alignment import align_targets_to_text\n"
                "from germeval_llm.schemas import TaskDTargetItem\n\n"
                "demo_text = 'Der ICE war pünktlich, aber der zweite ICE fiel wegen Schnee komplett aus.'\n"
                "demo_preds = [\n"
                "    TaskDTargetItem(target='ICE', category='Zugfahrt', polarity='positive'),\n"
                "    TaskDTargetItem(target='ICE', category='Zugfahrt', polarity='negative'),\n"
                "    TaskDTargetItem(target='fiel * aus', category='Zugfahrt', polarity='negative'),\n"
                "    TaskDTargetItem(target='Bordbistro', category='Gastronomisches_Angebot', polarity='negative'),\n"
                "]\n"
                "for sp in align_targets_to_text(demo_text, demo_preds):\n"
                "    slice_str = demo_text[sp.start_char:sp.end_char] if sp.start_char is not None else None\n"
                "    print(f'{sp.target:12s} | status={sp.alignment_status:20s} | span=({str(sp.start_char):>4s}, {str(sp.end_char):>4s}) -> {slice_str!r}')"
            ),
            new_markdown_cell(
                "### Key Takeaways\n"
                "- Prompting (`D1`) struggles with the GermEval discontinuous `' * '` convention (e.g., German verbal brackets), whereas QLoRA (`D2`) learns to emit `' * '`-joined targets reliably (**+19.7 F1 points** on `test_syn`).\n"
                "- Both String-level Multiset F1 and Explicit-Span Exact/Overlap F1 are computed in **100% pure Python** (`evaluate_task_d_predictions`), while also reconciling with `EvaluationScript.jar OTE`."
            ),
        ],
        "09_evaluation.ipynb": [
            new_markdown_cell(
                "# `09_evaluation.ipynb` — Pure-Python Evaluation & Official Java Reconciliation (85–105 min CAS Session)\n\n"
                "## 1. Pure-Python Evaluation Architecture (`src/germeval_llm/evaluation.py`)\n"
                "All metrics in this repository are computed in **pure Python** (no Java required at runtime on Google Colab) across two complementary scoring modes:\n\n"
                "1. **Strict Modern Set-Classification Metrics (`compute_set_classification_metrics`):**\n"
                "   - Evaluates **every document** in the split, including the ~16–18% `relevance=false` documents whose gold set is empty (`$\\emptyset$`).\n"
                "   - If a model hallucinates `Allgemein:negative` on an irrelevant post, it is properly penalized with `FP += 1`.\n"
                "   - Computes standard **Micro-Precision**, **Micro-Recall**, **Micro-F1** ($\\frac{2\\text{TP}}{2\\text{TP} + \\text{FP} + \\text{FN}}$), **Macro-F1**, **Exact-Set Accuracy**, and **paired bootstrap 95% confidence intervals** ($B=1{,}000$ resamples).\n\n"
                "2. **Exact Historical Java-Compatible Scorer (`evaluate_task_c_java_compatible`):**\n"
                "   To guarantee 100% comparability with 2017–2018 GermEval papers without needing Java 8 on Colab, we decompiled `EvaluationScript.jar` and replicated its two historical quirks in pure Python:\n"
                "   - **Quirk A — Empty-Gold Loop Skip (`GermevalEvaluator.java:186`):** The outer loop iterates over `for (String goldCat : goldCategories)`. When a document has 0 gold categories (`relevance=false`), the loop body never executes and any predicted labels on that document are silently ignored (`FP = 0`).\n"
                "   - **Quirk B — The `\"none\"` Placeholder Jaccard Bug (`Fscore.java:37–69`):** When recording a false negative `(gold, \"none\")` or false positive `(\"none\", pred)`, `getDistinctLabels(data)` includes `\"none\"` in the label set over which it sums `tp_sum`, `fp_sum`, and `fn_sum`. Because `\"none\"` never matches itself (`tp=0`), iterating over `\"none\"` adds all FNs to `fp_sum` and all FPs to `fn_sum`, **doubling `fp_sum + fn_sum`** and turning Java's `MICRO_F1` into the **Jaccard index**:\n"
                "     $$\\text{MICRO\\_F1}_{\\text{Java}} = \\frac{2 \\cdot \\text{TP}}{2 \\cdot \\text{TP} + 2(\\text{FP} + \\text{FN})} = \\frac{\\text{TP}}{\\text{TP} + \\text{FP} + \\text{FN}}$$"
            ),
            new_code_cell(SETUP_SNIPPET),
            new_code_cell(
                "bench = pd.read_csv('results/metrics/benchmark.csv')\n"
                "display(bench)\n\n"
                "rec = json.loads(Path('results/metrics/official_java_reconciliation.json').read_text())\n"
                "print('\\nPure-Python Java-Compat Scorer vs. Official EvaluationScript.jar (6-decimal agreement):')\n"
                "display(pd.DataFrame(rec['reconciliations']))"
            ),
            new_markdown_cell(
                "### Hands-on Demo: Running the Pure-Python Evaluator Directly on Any Prediction List\n"
                "Below we run `evaluate_task_c_predictions()` in pure Python on a 3-document synthetic example so you can see both `strict` and `java_compat` metrics computed side by side:"
            ),
            new_code_cell(
                "from germeval_llm.evaluation import evaluate_task_c_predictions\n"
                "from germeval_llm.xml_parser import load_canonical_jsonl\n"
                "from germeval_llm.inference import load_predictions_jsonl\n\n"
                "dev_docs_5 = load_canonical_jsonl('data/canonical/dev.jsonl')[:5]\n"
                "smoke_preds_5 = load_predictions_jsonl('results/predictions/smoke_lora_dev_dev.jsonl')\n"
                "live_eval = evaluate_task_c_predictions(dev_docs_5, smoke_preds_5, n_bootstrap=200)\n"
                "print('Pure-Python Evaluation Summary on 5-doc Smoke LoRA Run:')\n"
                "print(json.dumps({\n"
                "    'schema_validity_rate': live_eval['schema_validity_rate'],\n"
                "    'valid_exact_rate': live_eval['valid_exact_rate'],\n"
                "    'c1_strict_micro_f1': live_eval['task_c1_category_strict']['micro_f1'],\n"
                "    'c1_java_compat_micro_f1': live_eval['task_c1_category_java_compat']['micro_f1'],\n"
                "    'c2_strict_micro_f1': live_eval['task_c2_category_sentiment_strict']['micro_f1'],\n"
                "    'c2_java_compat_micro_f1': live_eval['task_c2_category_sentiment_java_compat']['micro_f1'],\n"
                "}, indent=2))"
            ),
            new_markdown_cell(
                "### Key Takeaways\n"
                "- Our pure-Python evaluator (`evaluation.py`) requires **zero external Java dependencies** on Google Colab while reproducing both modern Strict Micro/Macro-F1 and exact 6-decimal `EvaluationScript.jar` scores."
            ),
        ],
        "10_error_analysis.ipynb": [
            new_markdown_cell(
                "# `10_error_analysis.ipynb` — Linguistic & Systems Error Analysis (105–120 min CAS Session)\n\n"
                "## 1. Where Does QLoRA Beat 10-Shot Prompting?\n"
                "Aggregate Micro-F1 (`0.8360` for `Q1` QLoRA vs. `0.7052` for `P3` 10-shot on `test_syn`) tells us *that* fine-tuning wins, but not *why*. Below we decompose the gain along four linguistic and systems dimensions:\n"
                "1. **Rare-Category Recall vs. Support:** In a 10-shot prompt, only ~8–12 categories can appear in the demonstration examples. Categories with subtle boundaries (`Atmosphäre`, `Design`, `Gepäck`, `Barrierefreiheit`, `Auslastung_und_Platzangebot`) gain **+20 to +35 F1 points** from QLoRA.\n"
                "2. **Irrelevant Document (`relevance=false`) Precision:** Prompted LLMs suffer from *helpfulness bias*—when given a tweet mentioning *\"Bahnhof\"*, they try hard to assign `Allgemein:neutral` or `Image:negative` even when the tweet is not customer feedback about DB. Fine-tuning teaches the exact boundary of `{\"labels\": []}`.\n"
                "3. **Paired Document Contingency Quadrants (`McNemar` / Paired Win Analysis):** On `test_syn` (`2,566` docs), QLoRA gets the **entire document label set** right on **621 documents where `P3` failed**, while `P3` wins on only 113 documents.\n"
                "4. **Synchronic (`test_syn`) vs. Diachronic (`test_dia`) Drift:** Both prompting and fine-tuning experience a ~1.9–2.1 F1 drop on `test_TIMESTAMP2` (collected 6 months later during winter rail disruptions), showing that QLoRA adapts to the domain taxonomy without overfitting exclusively to summer 2016 topics."
            ),
            new_code_cell(SETUP_SNIPPET),
            new_code_cell(
                "display(Image(filename='results/figures/error_analysis_delta_f1.png'))\n\n"
                "quad = json.loads(Path('results/metrics/error_analysis_quadrants_test_syn.json').read_text())\n"
                "print('Paired Document Exact-Set Quadrants on test_syn (P3 10-shot vs Q1 QLoRA):')\n"
                "print(json.dumps({k: v for k, v in quad.items() if k != 'sample_qlora_wins' and k != 'sample_prompt_wins'}, indent=2))\n"
                "print('\\nSample documents where QLoRA succeeded and Prompting failed:')\n"
                "for ex in quad['sample_qlora_wins'][:3]:\n"
                "    print('-', ex['doc_id'], '| Gold:', ex['gold'], '| P3:', ex['prompt_pred'], '| QLoRA:', ex['qlora_pred'])"
            ),
            new_code_cell(
                "# Synchronic (test_syn) vs Diachronic (test_dia) generalization comparison\n"
                "bench = pd.read_csv('results/metrics/benchmark.csv')\n"
                "piv = bench[bench['run_id'].isin(['P1_guidelines_0shot', 'P3_dev_optimized_10shot', 'L1_lora_r16', 'Q1_qlora_r16'])].pivot(\n"
                "    index='run_id', columns='split', values='c2_strict_micro_f1'\n"
                ")[['dev', 'test_syn', 'test_dia']]\n"
                "piv['diachronic_drop (syn - dia)'] = (piv['test_syn'] - piv['test_dia']).round(4)\n"
                "display(piv)"
            ),
            new_markdown_cell(
                "### Final CAS Decision Matrix: When Should You Prompt vs. Fine-Tune?\n"
                "| Criterion | Guideline 0-Shot (`P1`) | 10-Shot ICL (`P3`) | 4-bit NF4 QLoRA (`Q1`) |\n"
                "|---|---|---|---|\n"
                "| **Labeled data required** | `0` docs | `10` docs | `$\\ge 250\\text{–}500$` docs (crossover) |\n"
                "| **Task C2 Strict Micro-F1 (`test_syn`)** | `0.644` | `0.705` | **`0.836`** (`+13.1` pts vs `P3`) |\n"
                "| **Task D Explicit-Span F1 (`test_syn`)** | `0.453` (`D1`) | `0.453` (`D1`) | **`0.650`** (`+19.7` pts vs `D1`) |\n"
                "| **Raw JSON Schema Compliance** | `~60–94%` (fences/drift) | `~98.8%` | **`99.9%+`** (`VALID_EXACT`) |\n"
                "| **Inference Prompt Tokens / 1k Docs** | `~550k` tokens | `~1,670k` tokens (**3× cost**) | **`~550k` tokens** (lean 0-shot prompt) |\n"
                "| **Peak Training GPU VRAM** | `0 GB` | `0 GB` | **`~2.9 GB`** (fits any T4 / L4 GPU) |"
            ),
        ],
        "11_colab_l4_live_demo.ipynb": [
            new_markdown_cell(
                "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simon-clematide/gemma-qlora-finetuning/blob/main/notebooks/11_colab_l4_live_demo.ipynb)\n\n"
                "# `11_colab_l4_live_demo.ipynb` — End-to-End Live QLoRA / LoRA Fine-Tuning & Test-Set Evaluation on Google Colab (L4 / T4 GPU)\n\n"
                "This self-contained notebook is designed to run **live on a Google Colab L4 GPU (`24 GB VRAM`) or free T4 GPU (`15 GB VRAM`)** (or locally on CUDA / Apple Silicon `mps`).\n"
                "- **Zero live XML preprocessing needed:** The repository includes all 26,209 canonical GermEval 2017 documents in compact `data/canonical/*.jsonl.gz` archives (`5 MB` total) and automatically materializes the Task C and Task D JSONL files in ~1.5 seconds.\n"
                "- **Configurable 4-bit NF4 QLoRA or 16-bit `bfloat16` LoRA:** Defaults to **4-bit NF4 QLoRA** (`USE_4BIT_QUANTIZATION = True` in Step 3), and can be switched to **16-bit `bfloat16` LoRA** (`USE_4BIT_QUANTIZATION = False`) with a single boolean toggle.\n"
                "- **100% Pure-Python Evaluation:** Computes Strict Task C1/C2 Micro-F1, Macro-F1, Exact-Set Accuracy, Bootstrap 95% CIs, and Exact Official Java-Compatible Scores in pure Python (`src/germeval_llm/evaluation.py`).\n\n"
                "---\n"
                "## Step 1: Google Colab Bootstrap (Clone Repo, Install Package, Authenticate HF)"
            ),
            new_code_cell(
                "import os, sys\n"
                "from pathlib import Path\n\n"
                "# If running on Google Colab and not yet inside the repository, clone it and install dependencies:\n"
                "if 'google.colab' in sys.modules:\n"
                "    REPO_URL = os.environ.get('GERMEVAL_REPO_URL', 'https://github.com/simon-clematide/gemma-qlora-finetuning.git')\n"
                "    if not Path('pyproject.toml').exists() and not Path('gemma-qlora-finetuning/pyproject.toml').exists():\n"
                "        print(f'Cloning {REPO_URL}...')\n"
                "        os.system(f'git clone {REPO_URL}')\n"
                "    if Path('gemma-qlora-finetuning/pyproject.toml').exists():\n"
                "        os.chdir('gemma-qlora-finetuning')\n"
                "    os.system('pip install -q -U \"bitsandbytes>=0.46.1\" \".[ml,cuda]\"')\n\n"
                "ROOT = Path.cwd().resolve()\n"
                "if ROOT.name == 'notebooks':\n"
                "    ROOT = ROOT.parent\n"
                "    os.chdir(ROOT)\n"
                "if str(ROOT / 'src') not in sys.path:\n"
                "    sys.path.insert(0, str(ROOT / 'src'))\n\n"
                "from germeval_llm.convert import ensure_derived_datasets_ready\n"
                "from germeval_llm.inference import detect_compute_device\n\n"
                "ensure_derived_datasets_ready()\n"
                "hw = detect_compute_device()\n"
                "print('Repository initialized:', ROOT.name)\n"
                "print('Detected compute hardware:', hw)"
            ),
            new_markdown_cell(
                "### Authenticate with Hugging Face Hub (for gated `google/gemma-2-2b-it`)\n"
                "Run `login(new_session=True)` below and paste your Hugging Face access token (`new_session=True` ensures Colab prompts you even if an older `HF_TOKEN` secret is cached in the runtime):"
            ),
            new_code_cell(
                "import os\n"
                "from huggingface_hub import login\n\n"
                "os.environ.pop('HF_TOKEN', None)\n"
                "os.environ.pop('HUGGING_FACE_HUB_TOKEN', None)\n"
                "login(new_session=True)"
            ),
            new_markdown_cell(
                "---\n"
                "## Step 2: Inspect the Dataset & Completion-Only Loss Masking\n"
                "Before loading the model, let's inspect a real training example and verify how our prompt and completion are formatted for `google/gemma-2-2b-it`. Remember:\n"
                "- **Task C Output Contract:** `{\"labels\": [\"Category:polarity\", ...]}` (or `{\"labels\": []}` when `relevance=false`).\n"
                "- **Completion-Only Loss (`labels = -100`):** During LoRA / QLoRA training, all prompt tokens up to `<start_of_turn>model\\n` are masked with `-100` so gradients only update on the JSON completion."
            ),
            new_code_cell(
                "import json\n"
                "import pandas as pd\n"
                "from IPython.display import display\n"
                "from germeval_llm.xml_parser import load_canonical_jsonl\n"
                "from germeval_llm.schemas import TaskCExample\n"
                "from germeval_llm.prompts import format_gemma_chat_string, compute_completion_only_labels\n\n"
                "train_docs = load_canonical_jsonl('data/canonical/train.jsonl')\n"
                "dev_docs = load_canonical_jsonl('data/canonical/dev.jsonl')\n"
                "test_syn_docs = load_canonical_jsonl('data/canonical/test_syn.jsonl')\n"
                "test_dia_docs = load_canonical_jsonl('data/canonical/test_dia.jsonl')\n"
                "print(f'Loaded splits -> train: {len(train_docs)}, dev: {len(dev_docs)}, test_syn: {len(test_syn_docs)}, test_dia: {len(test_dia_docs)}')\n\n"
                "with Path('data/derived/task_c/train.jsonl').open() as f:\n"
                "    sample_ex = TaskCExample.model_validate_json(f.readline())\n\n"
                "print('\\nSample Document Text:', sample_ex.text)\n"
                "print('Target JSON Completion:', sample_ex.completion[0].content)"
            ),
            new_markdown_cell(
                "---\n"
                "## Step 3: Configure Experiment Scope & Precision (`4-bit NF4 QLoRA` vs. `16-bit bfloat16 LoRA`)\n"
                "- **`USE_4BIT_QUANTIZATION = True` (Default — 4-bit `nf4` QLoRA via `configs/qlora.yaml`):**\n"
                "  Compresses frozen base weights by ~72% (`~1.4 GB` VRAM, `~2.9 GB` peak training VRAM) via `bitsandbytes` 4-bit NormalFloat quantization with double quantization (`bnb_4bit_use_double_quant=True`). Works on both Colab **L4 (`24 GB`)** and **T4 (`15 GB`)** GPUs.\n"
                "- **`USE_4BIT_QUANTIZATION = False` (16-bit `bfloat16` LoRA via `configs/lora.yaml`):**\n"
                "  Uses native 16-bit `bfloat16` base weights (`~5.0 GB` VRAM, `~7.2 GB` peak training VRAM). Requires no `bitsandbytes` dependency and trains ~20% faster per step on an L4 GPU.\n"
                "- **`RECOMPUTE_ZERO_SHOT_BASELINE = False` (Default — Instant Zero-Shot Baseline Reuse):**\n"
                "  Loads the stored Zero-Shot `P1_guidelines` predictions from `results/predictions/P1_guidelines_0shot_test_syn.jsonl` and scores them on `eval_docs` in <0.2 seconds so you can jump straight into Step 5 (`SFTTrainer` fine-tuning). Set `RECOMPUTE_ZERO_SHOT_BASELINE = True` if you want to run live Zero-Shot inference on the GPU."
            ),
            new_code_cell(
                "import torch\n\n"
                "# Set RUN_LIVE_GPU = True on Google Colab (automatically enabled when CUDA GPU is present!)\n"
                "RUN_LIVE_GPU = torch.cuda.is_available()\n"
                "USE_4BIT_QUANTIZATION = True           # True = 4-bit NF4 QLoRA (configs/qlora.yaml) | False = 16-bit bfloat16 LoRA (configs/lora.yaml)\n"
                "RECOMPUTE_ZERO_SHOT_BASELINE = False   # False = reuse stored results/predictions/P1_guidelines_0shot_test_syn.jsonl (instant!)\n"
                "DEMO_MODE = 'fast_l4'                  # Choose 'fast_l4' (~5 mins on L4) or 'full_l4' (~45 mins on L4)\n\n"
                "METHOD_TAG = 'qlora_4bit' if USE_4BIT_QUANTIZATION else 'lora_16bit'\n"
                "METHOD_LABEL = '4-bit NF4 QLoRA (r=16)' if USE_4BIT_QUANTIZATION else '16-bit bfloat16 LoRA (r=16)'\n"
                "TRAIN_CONFIG = 'configs/qlora.yaml' if USE_4BIT_QUANTIZATION else 'configs/lora.yaml'\n"
                "ADAPTER_DIR = f'models/adapters/colab_{METHOD_TAG}_task_c'\n"
                "STORED_BASELINE_PRED_PATH = 'results/predictions/P1_guidelines_0shot_test_syn.jsonl'\n"
                "LIVE_BASELINE_PRED_PATH = 'results/predictions/colab_baseline_p1_test_syn.jsonl'\n"
                "FT_PRED_PATH = f'results/predictions/colab_{METHOD_TAG}_r16_test_syn.jsonl'\n\n"
                "if DEMO_MODE == 'fast_l4':\n"
                "    TRAIN_SUBSET_SIZE = 1000\n"
                "    MAX_TRAIN_STEPS = 120\n"
                "    EVAL_MAX_DOCS = 150\n"
                "else:\n"
                "    TRAIN_SUBSET_SIZE = None  # All 19,432 train docs\n"
                "    MAX_TRAIN_STEPS = None    # Full 3 epochs (3,644 steps)\n"
                "    EVAL_MAX_DOCS = None      # All 2,566 test_syn docs\n\n"
                "print(f'RUN_LIVE_GPU={RUN_LIVE_GPU} | MODE={METHOD_LABEL} ({TRAIN_CONFIG}) | DEMO_MODE={DEMO_MODE}')\n"
                "print(f'RECOMPUTE_ZERO_SHOT_BASELINE={RECOMPUTE_ZERO_SHOT_BASELINE} | ADAPTER_DIR={ADAPTER_DIR}')"
            ),
            new_markdown_cell(
                "---\n"
                "## Step 4: Baseline Evaluation on `test_syn` (`test_TIMESTAMP1`) Before Fine-Tuning\n"
                "By default (`RECOMPUTE_ZERO_SHOT_BASELINE = False`), we load the precomputed Zero-Shot (`P1_guidelines`) predictions on `test_syn` from `results/predictions/P1_guidelines_0shot_test_syn.jsonl` and score the exact `eval_docs` slice with our **pure-Python evaluator** (`evaluate_task_c_predictions`) in <0.2 seconds—saving several minutes of GPU inference before training. (Set `RECOMPUTE_ZERO_SHOT_BASELINE = True` in Step 3 if you want to recompute them live on the GPU.)"
            ),
            new_code_cell(
                "from germeval_llm.inference import (\n"
                "    load_gemma_model_and_tokenizer,\n"
                "    build_hf_generator_fn,\n"
                "    run_task_c_inference_with_generator,\n"
                "    save_predictions_jsonl,\n"
                "    load_predictions_jsonl,\n"
                ")\n"
                "from germeval_llm.evaluation import evaluate_task_c_predictions\n\n"
                "eval_docs = test_syn_docs[:EVAL_MAX_DOCS] if EVAL_MAX_DOCS else test_syn_docs\n\n"
                "if RUN_LIVE_GPU and RECOMPUTE_ZERO_SHOT_BASELINE:\n"
                "    print('Running live Zero-Shot P1 inference on GPU...')\n"
                "    base_model, tokenizer = load_gemma_model_and_tokenizer(\n"
                "        model_id='google/gemma-2-2b-it',\n"
                "        load_in_4bit=USE_4BIT_QUANTIZATION,\n"
                "    )\n"
                "    gen_fn = build_hf_generator_fn(base_model, tokenizer, max_new_tokens=256)\n"
                "    baseline_preds = run_task_c_inference_with_generator(\n"
                "        eval_docs,\n"
                "        run_id='colab_baseline_p1',\n"
                "        split='test_syn',\n"
                "        generate_fn=gen_fn,\n"
                "        prompt_key='p1_guidelines',\n"
                "    )\n"
                "    save_predictions_jsonl(baseline_preds, LIVE_BASELINE_PRED_PATH)\n"
                "    baseline_metrics = evaluate_task_c_predictions(eval_docs, baseline_preds)\n"
                "    del base_model\n"
                "    torch.cuda.empty_cache()\n"
                "else:\n"
                "    print(f'Reusing stored Zero-Shot P1 predictions from {STORED_BASELINE_PRED_PATH} ({len(eval_docs)} docs)...')\n"
                "    stored_preds_all = load_predictions_jsonl(STORED_BASELINE_PRED_PATH)\n"
                "    baseline_preds = stored_preds_all[:len(eval_docs)]\n"
                "    baseline_metrics = evaluate_task_c_predictions(eval_docs, baseline_preds)\n\n"
                "print('Baseline (Zero-Shot P1) Summary:')\n"
                "print(json.dumps({\n"
                "    'num_documents': baseline_metrics['num_documents'],\n"
                "    'schema_validity_rate': baseline_metrics['schema_validity_rate'],\n"
                "    'valid_exact_rate': baseline_metrics['valid_exact_rate'],\n"
                "    'parse_status_counts': baseline_metrics['parse_status_counts'],\n"
                "    'c1_strict_micro_f1': round(baseline_metrics['task_c1_category_strict']['micro_f1'], 4),\n"
                "    'c2_strict_micro_f1': round(baseline_metrics['task_c2_category_sentiment_strict']['micro_f1'], 4),\n"
                "    'c2_java_compat_micro_f1': round(baseline_metrics['task_c2_category_sentiment_java_compat']['micro_f1'], 4),\n"
                "}, indent=2))"
            ),
            new_markdown_cell(
                "---\n"
                "## Step 5: Live QLoRA / LoRA Fine-Tuning (`trl.SFTTrainer`)\n"
                "Now we train our PEFT adapter ($r=16, \\alpha=32$, targeting all 7 linear projections `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`) using Hugging Face TRL's **`SFTTrainer`** (`Supervised Fine-Tuning Trainer`) with `completion_only_loss=True`.\n\n"
                "### What Does `SFTTrainer` + QLoRA / LoRA Modify?\n"
                "1. **Inside Each Linear Layer (Frozen Base $W_0$ + 16-bit LoRA Bypass):**\n"
                "```text\n"
                "                        Output Activation h = h_base + (α/r) · h_lora\n"
                "                                       ▲\n"
                "                                       │\n"
                "                                      (+)  ◄── Elementwise Sum in bfloat16\n"
                "                                     ╱   ╲\n"
                "         ┌──────────────────────────┘     └───────────────────────────┐\n"
                "         │  h_base = W_0 · x                h_lora = (α/r) · B · (A · x)\n"
                "┌────────┴────────────────────────┐          ┌────────────────────────┴────────┐\n"
                "│   FROZEN BASE WEIGHT W_0        │          │   TRAINABLE LoRA ADAPTER (B·A)  │\n"
                "│ • 4-bit NF4   (~1.4 GB) OR      │          │ • Matrix B [d_out × r=16] (bf16)│\n"
                "│   16-bit bf16 (~5.0 GB)         │          │ • Matrix A [r=16 × d_in]  (bf16)│\n"
                "│ • requires_grad = False         │          │ • requires_grad = True          │\n"
                "│ • 0 optimizer states            │          │ • Updated by SFTTrainer AdamW   │\n"
                "└────────▲────────────────────────┘          └────────────────▲────────────────┘\n"
                "         └──────────────────────────┐     ┌───────────────────┘\n"
                "                                     ╲   ╱\n"
                "                             Input Activation x (bfloat16)\n"
                "```\n\n"
                "2. **Inside Each of the 26 Gemma-2-2B Transformer Blocks (7 Modified Linear Projections):**\n"
                "```text\n"
                "  Input Tokens ──► [ Token Embeddings (256,000 × 2304) ]  ──► FROZEN\n"
                "                                │\n"
                "        ┌───────────────────────▼───────────────────────┐\n"
                "        │           LAYER l ∈ {1 ... 26}                │\n"
                "        │  • Grouped-Query Attention (GQA):             │\n"
                "        │     ├─► q_proj (2304 → 2048)  [+LoRA A, B] ◄──┼── MODIFIED (r=16)\n"
                "        │     ├─► k_proj (2304 → 1024)  [+LoRA A, B] ◄──┼── MODIFIED (r=16)\n"
                "        │     ├─► v_proj (2304 → 1024)  [+LoRA A, B] ◄──┼── MODIFIED (r=16)\n"
                "        │     └─► o_proj (2048 → 2304)  [+LoRA A, B] ◄──┼── MODIFIED (r=16)\n"
                "        │  • GeGLU MLP (Feed-Forward Network):          │\n"
                "        │     ├─► gate_proj (2304 → 9216) [+LoRA A, B] ◄┼── MODIFIED (r=16)\n"
                "        │     ├─► up_proj   (2304 → 9216) [+LoRA A, B] ◄┼── MODIFIED (r=16)\n"
                "        │     └─► down_proj (9216 → 2304) [+LoRA A, B] ◄┼── MODIFIED (r=16)\n"
                "        │  • All RMSNorms & Residuals ──────────────────┼──► FROZEN\n"
                "        └───────────────────────┬───────────────────────┘\n"
                "  TOTAL BASE PARAMETERS (FROZEN): 2,614,341,888 (99.21%)\n"
                "  TOTAL LoRA PARAMETERS (TRAIN) :    20,766,720 ( 0.79%) = 798,720 × 26 layers\n"
                "```\n\n"
                "3. **Across the Token Sequence (`completion_only_loss=True` Masking):**\n"
                "```text\n"
                "Token Stream:   <bos> <start_of_turn>user\\n ... [~860 Prompt Tokens] ... <start_of_turn>model\\n {\"labels\": [\"Zugfahrt:negative\"]} <end_of_turn>\\n\n"
                "Target Labels:  -100         -100                        -100                            -100   \"{\"  \"labels\" ... \"Zugfahrt:negative\" \"]}\" <end_of_turn>\n"
                "                  └───────────────────────────────────┬───────────────────────────────────┘      └────────────────────┬────────────────────┘\n"
                "                                       IGNORED BY CROSS-ENTROPY LOSS                                   ACTIVE CROSS-ENTROPY LOSS\n"
                "                                         (No Gradient Signal)                                        (100% of Backprop Updates A & B)\n"
                "```"
            ),
            new_code_cell(
                "from germeval_llm.training import run_sft_adapter_training\n\n"
                "if RUN_LIVE_GPU:\n"
                "    train_summary = run_sft_adapter_training(\n"
                "        config_path=TRAIN_CONFIG,\n"
                "        train_jsonl_path='data/derived/task_c/train.jsonl',\n"
                "        dev_jsonl_path='data/derived/task_c/dev.jsonl',\n"
                "        output_adapter_dir=ADAPTER_DIR,\n"
                "        subset_size=TRAIN_SUBSET_SIZE,\n"
                "        max_steps_override=MAX_TRAIN_STEPS,\n"
                "        sampling_strategy='stratified',\n"
                "        seed_override=42,\n"
                "    )\n"
                "    torch.cuda.empty_cache()\n"
                "else:\n"
                "    train_summary = json.loads(Path('results/runs/smoke_task_c_lora_train_summary.json').read_text())\n\n"
                "print(f'Adapter Training Summary ({METHOD_LABEL}):')\n"
                "print(json.dumps(train_summary, indent=2))"
            ),
            new_markdown_cell(
                "---\n"
                "## Step 6: Evaluate the Fine-Tuned Adapter on `test_syn` & Compare Side-by-Side\n"
                "Now we reload the saved adapter (`ADAPTER_DIR`) onto the base model using the same precision setting (`load_in_4bit=USE_4BIT_QUANTIZATION`), run greedy inference on the exact same `test_syn` split, and compare Before vs. After metrics in pure Python:"
            ),
            new_code_cell(
                "if RUN_LIVE_GPU:\n"
                "    ft_model, tokenizer = load_gemma_model_and_tokenizer(\n"
                "        model_id='google/gemma-2-2b-it',\n"
                "        adapter_path=ADAPTER_DIR,\n"
                "        load_in_4bit=USE_4BIT_QUANTIZATION,\n"
                "    )\n"
                "    ft_gen_fn = build_hf_generator_fn(ft_model, tokenizer, max_new_tokens=256)\n"
                "    ft_preds = run_task_c_inference_with_generator(\n"
                "        eval_docs,\n"
                "        run_id=f'colab_{METHOD_TAG}_r16',\n"
                "        split='test_syn',\n"
                "        generate_fn=ft_gen_fn,\n"
                "        prompt_key='p1_guidelines',\n"
                "    )\n"
                "    save_predictions_jsonl(ft_preds, FT_PRED_PATH)\n"
                "    ft_metrics = evaluate_task_c_predictions(eval_docs, ft_preds)\n"
                "    del ft_model\n"
                "    torch.cuda.empty_cache()\n"
                "else:\n"
                "    ft_preds = load_predictions_jsonl('results/predictions/smoke_lora_dev_dev.jsonl')\n"
                "    ft_metrics = json.loads(Path('results/metrics/smoke_lora_dev_dev.json').read_text())\n\n"
                "comparison_df = pd.DataFrame([\n"
                "    {\n"
                "        'Model': '1. Frozen Gemma-2-2B-IT (Zero-Shot P1)',\n"
                "        'Valid Exact JSON Rate': baseline_metrics['valid_exact_rate'],\n"
                "        'Schema Validity Rate': baseline_metrics['schema_validity_rate'],\n"
                "        'Task C1 Strict Micro-F1': round(baseline_metrics['task_c1_category_strict']['micro_f1'], 4),\n"
                "        'Task C2 Strict Micro-F1': round(baseline_metrics['task_c2_category_sentiment_strict']['micro_f1'], 4),\n"
                "        'Task C2 Official Java-Compat F1': round(baseline_metrics['task_c2_category_sentiment_java_compat']['micro_f1'], 4),\n"
                "        'Mean Latency (ms/doc)': round(baseline_metrics['efficiency']['mean_latency_ms_per_doc'], 1),\n"
                "    },\n"
                "    {\n"
                "        'Model': f'2. Fine-Tuned Gemma-2-2B-IT ({METHOD_LABEL})',\n"
                "        'Valid Exact JSON Rate': ft_metrics['valid_exact_rate'],\n"
                "        'Schema Validity Rate': ft_metrics['schema_validity_rate'],\n"
                "        'Task C1 Strict Micro-F1': round(ft_metrics['task_c1_category_strict']['micro_f1'], 4),\n"
                "        'Task C2 Strict Micro-F1': round(ft_metrics['task_c2_category_sentiment_strict']['micro_f1'], 4),\n"
                "        'Task C2 Official Java-Compat F1': round(ft_metrics['task_c2_category_sentiment_java_compat']['micro_f1'], 4),\n"
                "        'Mean Latency (ms/doc)': round(ft_metrics['efficiency']['mean_latency_ms_per_doc'], 1),\n"
                "    },\n"
                "])\n"
                "display(comparison_df)"
            ),
            new_markdown_cell(
                "---\n"
                "## Step 7: Qualitative Inspection of Predictions Before vs. After Fine-Tuning\n"
                "Let's inspect the raw strings emitted by the base model vs. the fine-tuned adapter on individual documents:"
            ),
            new_code_cell(
                "from germeval_llm.convert import extract_task_c_label_sets\n\n"
                "docs_slice = eval_docs[:5] if RUN_LIVE_GPU else dev_docs[:5]\n"
                "b_slice = baseline_preds[:5]\n"
                "q_slice = ft_preds[:5]\n\n"
                "for doc, bp, qp in zip(docs_slice, b_slice, q_slice):\n"
                "    gold_c2, _ = extract_task_c_label_sets(doc, apply_typo_patches=True)\n"
                "    print('=' * 90)\n"
                "    print('Text:', doc.text[:160])\n"
                "    print('Gold C2 Labels :', gold_c2)\n"
                "    print(f'Zero-Shot Raw  ({bp.parse_status}): {bp.raw_output!r}')\n"
                "    print(f'Fine-Tuned Raw ({qp.parse_status}): {qp.raw_output!r}')"
            ),
        ],
    }


def build_and_execute_all_notebooks(
    notebooks_dir: str | Path = "notebooks",
    execute_cells: bool = False,
) -> None:
    out_dir = Path(notebooks_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = get_notebooks_spec()

    for nb_name, cells in specs.items():
        nb = _build_notebook(cells)
        nb_path = out_dir / nb_name
        if execute_cells:
            client = nbclient.NotebookClient(nb, timeout=180, kernel_name="python3")
            client.execute()
        nbformat.write(nb, str(nb_path))
        print(f"Generated (execute_cells={execute_cells}): {nb_path}")


if __name__ == "__main__":
    import sys

    exec_flag = "--execute" in sys.argv
    build_and_execute_all_notebooks(execute_cells=exec_flag)

