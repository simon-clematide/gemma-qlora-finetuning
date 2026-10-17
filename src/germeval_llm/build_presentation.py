"""Generate a clean, professional PowerPoint presentation (.pptx) introducing the core concepts
of our GermEval 2017 Gemma-2-2B-IT fine-tuning study:
1. Data & Task Formulation (GermEval 2017 ABSA -> JSON SFT + Completion-Only Loss Masking)
2. Low-Rank Adapters (LoRA Architecture, Modified Projections, Parameter Math)
3. Quantization (4-bit NF4 QLoRA vs. 16-bit bfloat16 LoRA, VRAM & Empirical Results)
"""

from __future__ import annotations

from pathlib import Path
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


# Color Palette (Modern Academic Navy / Slate / Teal / Amber)
NAVY = RGBColor(15, 23, 42)         # #0F172A - Headers / Dark accents
BLUE = RGBColor(37, 99, 235)        # #2563EB - Primary accent
TEAL = RGBColor(13, 148, 136)       # #0D9488 - Secondary / LoRA trainable accent
AMBER = RGBColor(217, 119, 6)       # #D97706 - Quantization / Highlight accent
SLATE_DARK = RGBColor(51, 65, 85)   # #334155 - Body text
SLATE_MUTED = RGBColor(100, 116, 139) # #64748B - Subtitles / Captions
BG_LIGHT = RGBColor(248, 250, 252)  # #F8FAFC - Card background
BG_BLUE = RGBColor(239, 246, 255)   # #EFF6FF - Soft blue card
BG_TEAL = RGBColor(240, 253, 250)   # #F0FDFA - Soft teal card
BG_AMBER = RGBColor(255, 251, 235)  # #FFFBEB - Soft amber card
BORDER_GRAY = RGBColor(203, 213, 225) # #CBD5E1 - Card border
WHITE = RGBColor(255, 255, 255)


def add_header(slide, title_text: str, subtitle_text: str, slide_num: int, total_slides: int) -> None:
    """Add a consistent top banner with title, subtitle, and slide number."""
    # Top accent bar
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.12))
    bar.fill.solid()
    bar.fill.fore_color.rgb = BLUE
    bar.line.fill.background()

    # Title box
    tb = slide.shapes.add_textbox(Inches(0.6), Inches(0.25), Inches(11.2), Inches(0.85))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p = tf.paragraphs[0]
    p.text = title_text
    p.font.size = Pt(26)
    p.font.bold = True
    p.font.color.rgb = NAVY
    p.font.name = "Calibri"

    p2 = tf.add_paragraph()
    p2.text = subtitle_text
    p2.font.size = Pt(13.5)
    p2.font.color.rgb = SLATE_MUTED
    p2.font.name = "Calibri"

    # Slide counter in top right
    num_box = slide.shapes.add_textbox(Inches(11.8), Inches(0.3), Inches(1.0), Inches(0.4))
    ntf = num_box.text_frame
    np = ntf.paragraphs[0]
    np.text = f"{slide_num} / {total_slides}"
    np.alignment = PP_ALIGN.RIGHT
    np.font.size = Pt(12)
    np.font.bold = True
    np.font.color.rgb = SLATE_MUTED
    np.font.name = "Calibri"

    # Divider line
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(1.18), Inches(12.133), Inches(0.02))
    line.fill.solid()
    line.fill.fore_color.rgb = BORDER_GRAY
    line.line.fill.background()


def add_card(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    title: str,
    bullets: list[str | tuple[str, str]],
    bg_color: RGBColor = BG_LIGHT,
    accent_color: RGBColor = BLUE,
    title_size: int = 16,
    body_size: int = 13,
) -> None:
    """Add a styled card with an accent header strip and bullet points."""
    card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    card.adjustments[0] = 0.02
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    card.line.color.rgb = BORDER_GRAY
    card.line.width = Pt(1)

    # Left vertical accent strip
    strip = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(left), Inches(top + 0.08), Inches(0.08), Inches(height - 0.16)
    )
    strip.fill.solid()
    strip.fill.fore_color.rgb = accent_color
    strip.line.fill.background()

    tb = slide.shapes.add_textbox(
        Inches(left + 0.22), Inches(top + 0.14), Inches(width - 0.35), Inches(height - 0.24)
    )
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p_title = tf.paragraphs[0]
    p_title.text = title
    p_title.font.size = Pt(title_size)
    p_title.font.bold = True
    p_title.font.color.rgb = accent_color
    p_title.font.name = "Calibri"
    p_title.space_after = Pt(6)

    for item in bullets:
        p = tf.add_paragraph()
        p.space_after = Pt(4)
        p.font.name = "Calibri"
        if isinstance(item, tuple):
            bold_prefix, rest = item
            r1 = p.add_run()
            r1.text = f"• {bold_prefix} "
            r1.font.bold = True
            r1.font.size = Pt(body_size)
            r1.font.color.rgb = NAVY
            r1.font.name = "Calibri"

            r2 = p.add_run()
            r2.text = rest
            r2.font.bold = False
            r2.font.size = Pt(body_size)
            r2.font.color.rgb = SLATE_DARK
            r2.font.name = "Calibri"
        else:
            p.text = f"• {item}"
            p.font.size = Pt(body_size)
            p.font.color.rgb = SLATE_DARK


def add_code_box(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    title: str,
    code_lines: list[str],
    accent_color: RGBColor = NAVY,
    font_size: int = 11,
) -> None:
    """Add a dark-themed monospace box for JSON schemas, token streams, or architecture diagrams."""
    box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    box.adjustments[0] = 0.02
    box.fill.solid()
    box.fill.fore_color.rgb = NAVY
    box.line.color.rgb = accent_color
    box.line.width = Pt(1.5)

    tb = slide.shapes.add_textbox(
        Inches(left + 0.18), Inches(top + 0.12), Inches(width - 0.36), Inches(height - 0.24)
    )
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p0 = tf.paragraphs[0]
    p0.text = title
    p0.font.size = Pt(12.5)
    p0.font.bold = True
    p0.font.color.rgb = RGBColor(96, 165, 250)  # Light blue
    p0.font.name = "Calibri"
    p0.space_after = Pt(4)

    for line in code_lines:
        p = tf.add_paragraph()
        p.text = line
        p.font.size = Pt(font_size)
        p.font.color.rgb = RGBColor(226, 232, 240)
        p.font.name = "Courier New"
        p.space_after = Pt(1)


def build_presentation(output_path: str | Path = "docs/gemma_qlora_core_concepts.pptx") -> Path:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]
    total_slides = 8

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = NAVY
    bg1.line.fill.background()

    # Accent stripe on left
    stripe1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.7), Inches(1.3), Inches(0.12), Inches(4.8))
    stripe1.fill.solid()
    stripe1.fill.fore_color.rgb = TEAL
    stripe1.line.fill.background()

    tb1 = s1.shapes.add_textbox(Inches(1.1), Inches(1.3), Inches(11.3), Inches(4.8))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p = tf1.paragraphs[0]
    p.text = "PARAMETER-EFFICIENT LLM ADAPTATION"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = RGBColor(45, 212, 191)
    p.font.name = "Calibri"
    p.space_after = Pt(10)

    p2 = tf1.add_paragraph()
    p2.text = "Fine-Tuning Gemma-2-2B-IT with LoRA & 4-bit QLoRA"
    p2.font.size = Pt(36)
    p2.font.bold = True
    p2.font.color.rgb = WHITE
    p2.font.name = "Calibri"
    p2.space_after = Pt(8)

    p3 = tf1.add_paragraph()
    p3.text = "Core Concepts: Instruction Data, Low-Rank Adapters (LoRA), and 4-bit NF4 Quantization on GermEval 2017"
    p3.font.size = Pt(18)
    p3.font.color.rgb = RGBColor(148, 163, 184)
    p3.font.name = "Calibri"
    p3.space_after = Pt(28)

    # 3 Pillar summary pills on title slide
    pillars = [
        ("1. DATA & LOSS MASKING", "26,209 German Customer Reviews\nJSON Schema + Completion-Only Loss", BLUE),
        ("2. LOW-RANK ADAPTERS (LoRA)", "Freeze 2.61B Base Weights (99.21%)\nTrain Only 20.8M Rank-16 Weights (0.79%)", TEAL),
        ("3. 4-BIT NF4 QUANTIZATION (QLoRA)", "72% Base Weight Compression (1.4 GB)\nRuns on Free Colab T4 / L4 GPUs", AMBER),
    ]
    for idx, (ptitle, pdesc, pcol) in enumerate(pillars):
        x = 1.1 + idx * 3.85
        box = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(4.45), Inches(3.6), Inches(1.65))
        box.adjustments[0] = 0.04
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(30, 41, 59)
        box.line.color.rgb = pcol
        box.line.width = Pt(1.5)

        btf = box.text_frame
        btf.word_wrap = True
        btf.margin_left = btf.margin_right = Inches(0.18)
        btf.margin_top = Inches(0.16)

        bp1 = btf.paragraphs[0]
        bp1.text = ptitle
        bp1.font.size = Pt(12)
        bp1.font.bold = True
        bp1.font.color.rgb = pcol
        bp1.font.name = "Calibri"
        bp1.space_after = Pt(6)

        bp2 = btf.add_paragraph()
        bp2.text = pdesc
        bp2.font.size = Pt(12.5)
        bp2.font.color.rgb = WHITE
        bp2.font.name = "Calibri"

    # =========================================================================
    # SLIDE 2: MOTIVATION — WHY FINE-TUNE INSTEAD OF JUST PROMPTING?
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(
        s2,
        "1. Motivation: Prompting vs. Parameter-Efficient Fine-Tuning (PEFT)",
        "Why adapt a compact 2.6B open-weights model (google/gemma-2-2b-it) instead of relying on Zero/Few-Shot prompting?",
        2,
        total_slides,
    )

    add_card(
        s2,
        left=0.6,
        top=1.4,
        width=5.9,
        height=2.65,
        title="The Limits of In-Context Prompting (Zero / Few-Shot)",
        bullets=[
            ("Taxonomy Ambiguity:", "19 domain-specific German aspect categories (Allgemein, Zugfahrt, Pünktlichkeit, Sicherheit...) require exact boundary rules."),
            ("Multi-Label Calibration:", "Zero-shot models over-predict or under-predict empty sets (relevance=false -> {\"labels\": []})."),
            ("Inference Token Overhead:", "10-shot prompting prepends ~1,500 tokens to EVERY single test document, doubling latency and KV-cache memory."),
        ],
        bg_color=BG_LIGHT,
        accent_color=SLATE_DARK,
    )

    add_card(
        s2,
        left=6.8,
        top=1.4,
        width=5.9,
        height=2.65,
        title="Why Full Fine-Tuning Is Impractical for LLMs",
        bullets=[
            ("High Memory Footprint:", "Updating all 2.61B parameters in 16-bit AdamW requires ~5.2 GB weights + ~5.2 GB gradients + ~20.8 GB optimizer states (>32 GB VRAM)."),
            ("Catastrophic Forgetting:", "Overwriting all base weights on a narrow classification schema degrades general language understanding."),
            ("Storage Cost:", "Saving 10 experiment checkpoints costs ~52 GB of disk space instead of ~80 MB per adapter."),
        ],
        bg_color=BG_AMBER,
        accent_color=AMBER,
    )

    add_card(
        s2,
        left=0.6,
        top=4.25,
        width=12.1,
        height=2.75,
        title="The Solution: Supervised Fine-Tuning (SFT) with LoRA & QLoRA",
        bullets=[
            ("Task-Specific Calibration:", "Teaches exact German annotation boundaries and guaranteed JSON compliance (100.0% valid JSON on test_syn)."),
            ("Massive Accuracy Jump:", "Boosts Task C2 Micro-F1 from 0.258 (Zero-Shot) and 0.453 (10-Shot) up to 0.720 (LoRA / QLoRA) — a +26.7 F1 gain over 10-shot."),
            ("Zero Inference Overhead:", "Trained low-rank matrices (B·A) can be merged directly into W_0 or run with compact Zero-Shot prompts (2.1x faster than 10-shot)."),
            ("Single-GPU Accessibility:", "Trains in ~2.9 GB peak VRAM with 4-bit NF4 QLoRA or ~7.2 GB with 16-bit bfloat16 LoRA on a single Google Colab GPU."),
        ],
        bg_color=BG_TEAL,
        accent_color=TEAL,
    )

    # =========================================================================
    # SLIDE 3: CONCEPT 1 (PART A) — THE DATA & TASK FORMULATION
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(
        s3,
        "2. Core Concept I — Data: GermEval 2017 ABSA as Structured JSON Generation",
        "Converting 26,209 German Deutsche Bahn customer feedback documents into a deterministic instruction-tuning contract",
        3,
        total_slides,
    )

    add_card(
        s3,
        left=0.6,
        top=1.4,
        width=5.8,
        height=5.6,
        title="Dataset & Task Formulation",
        bullets=[
            ("Corpus Splits (26,209 Docs):", "Train: 19,432 | Dev: 2,369 | Test Synchronic: 2,566 | Test Diachronic: 1,842 social media & web comments."),
            ("Task C (Aspect–Polarity Classification):", "Predict all unique (Category:polarity) pairs expressed in the document across 19 aspect categories × 3 polarities (positive, negative, neutral)."),
            ("Implicit Relevance Handling:", "Out-of-domain documents (relevance=false, ~21% of corpus) map cleanly to an empty JSON list: {\"labels\": []}."),
            ("Task D (Opinion Target Extraction):", "Extract exact character-grounded target phrases (OTE) for non-Allgemein opinions alongside category and polarity."),
            ("Canonical Deduplication & Sorting:", "Target JSON lists are sorted alphabetically so every document has a unique, deterministic gold target string."),
            ("Zero-Preprocessing Colab Bootstrap:", "All 26,209 canonical records ship as 5 MB of .jsonl.gz archives and materialize derived SFT splits in 1.5 seconds."),
        ],
        bg_color=BG_BLUE,
        accent_color=BLUE,
    )

    add_code_box(
        s3,
        left=6.65,
        top=1.4,
        width=6.05,
        height=2.65,
        title="Raw GermEval 2017 XML Input Document",
        code_lines=[
            '<Document id="http://twitter.com/.../697803">',
            '  <relevance>true</relevance>',
            '  <sentiment>negative</sentiment>',
            '  <text>@DB_Bahn Mal wieder 45 Minuten Verspätung',
            '        im ICE nach München und das WLAN geht',
            '        auch nicht!</text>',
            '  <Opinions>',
            '    <Opinion category="Zugfahrt#Pünktlichkeit"',
            '             polarity="negative" target="Verspätung"/>',
            '    <Opinion category="Sonstige_Unregelmässigkeiten"',
            '             polarity="negative" target="WLAN"/>',
            '  </Opinions>',
            '</Document>',
        ],
        accent_color=BLUE,
        font_size=10.5,
    )

    add_code_box(
        s3,
        left=6.65,
        top=4.25,
        width=6.05,
        height=2.75,
        title="Derived Task C & Task D JSON Completion Targets",
        code_lines=[
            "# Task C Target Completion (Sorted Set of Category:polarity):",
            '{',
            '  "labels": [',
            '    "Sonstige_Unregelmässigkeiten:negative",',
            '    "Zugfahrt:negative"',
            '  ]',
            '}',
            '',
            "# Task D Target Completion (Explicit Target Spans):",
            '{',
            '  "targets": [',
            '    {"category": "Zugfahrt", "polarity": "negative",',
            '     "target": "Verspätung"},',
            '    {"category": "Sonstige_Unregelmässigkeiten",',
            '     "polarity": "negative", "target": "WLAN"}',
            '  ]',
            '}',
        ],
        accent_color=TEAL,
        font_size=10.0,
    )

    # =========================================================================
    # SLIDE 4: CONCEPT 1 (PART B) — GEMMA-2 CHAT TEMPLATE & COMPLETION-ONLY LOSS
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(
        s4,
        "3. Core Concept I — Data: Chat Template & Completion-Only Loss Masking",
        "Why masking prompt tokens with -100 in SFTTrainer is essential when prompts are ~50x longer than JSON completions",
        4,
        total_slides,
    )

    add_code_box(
        s4,
        left=0.6,
        top=1.4,
        width=12.1,
        height=2.55,
        title="Token Stream & Label Masking Inside trl.SFTTrainer (completion_only_loss=True)",
        code_lines=[
            "Input IDs :  <bos> <start_of_turn>user\\n [ ~860 Guideline & Review Tokens ] <end_of_turn>\\n <start_of_turn>model\\n {\"labels\": [\"Zugfahrt:negative\"]} <end_of_turn>\\n",
            "Labels    :  -100       -100                             -100                    -100            -100        -100   \"{\"  \"labels\" ... \"Zugfahrt:negative\" \"]}\" <end_of_turn>",
            "             └───────────────────────────────────────────────────┬───────────────────────────────────────────────┘  └──────────────────────┬──────────────────────┘",
            "                                      MASKED OUT (PyTorch ignore_index = -100)                                            ACTIVE CROSS-ENTROPY LOSS (~15-25 Tokens)",
            "                                    0% of Gradient Signal (Never memorize prompt!)                                     100% of Backprop Updates LoRA Matrices A & B",
        ],
        accent_color=TEAL,
        font_size=10.5,
    )

    add_card(
        s4,
        left=0.6,
        top=4.2,
        width=5.9,
        height=2.8,
        title="Gemma-2-2B-IT Native Turn Formatting",
        bullets=[
            ("No Separate System Role:", "Gemma-2 only supports alternating user and model turns wrapped in <start_of_turn> and <end_of_turn>."),
            ("Unified Instruction Block:", "We merge the German task guidelines, the 19 category definitions, and the input review into the user turn."),
            ("Train-Test Parity:", "The exact same prompt wrapper (format_gemma_chat_string) is used during Zero-Shot evaluation and SFT training."),
        ],
        bg_color=BG_LIGHT,
        accent_color=BLUE,
    )

    add_card(
        s4,
        left=6.8,
        top=4.2,
        width=5.9,
        height=2.8,
        title="Why Completion-Only Loss Matters",
        bullets=[
            ("50:1 Length Asymmetry:", "The German guideline prompt is ~860 tokens long, while the target JSON output is only ~15–25 tokens."),
            ("Without Masking:", "97% of gradient updates would be wasted teaching the model to predict our static German instructions."),
            ("With Masking (labels = -100):", "PyTorch CrossEntropyLoss ignores all positions with label -100, focusing 100% of adapter capacity on JSON classification."),
        ],
        bg_color=BG_TEAL,
        accent_color=TEAL,
    )

    # =========================================================================
    # SLIDE 5: CONCEPT 2 — LOW-RANK ADAPTERS (LoRA ARCHITECTURE & MATH)
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(
        s5,
        "4. Core Concept II — Low-Rank Adaptation (LoRA) Architecture",
        "Injecting trainable rank-16 bottleneck matrices (A, B) in parallel with frozen pretrained weights W_0",
        5,
        total_slides,
    )

    add_code_box(
        s5,
        left=0.6,
        top=1.4,
        width=6.3,
        height=5.6,
        title="Inside Every Linear Projection Layer: Frozen W_0 + LoRA Bypass",
        code_lines=[
            "              Output Activation: h = h_base + (alpha / r) * h_lora",
            "                                     ▲",
            "                                     │",
            "                                    (+)  ◄── Elementwise Sum (bfloat16)",
            "                                   ╱   ╲",
            "       ┌──────────────────────────┘     └─────────────────────────┐",
            "       │  h_base = W_0 * x               h_lora = B * (A * x)     │",
            "┌──────┴──────────────────────┐           ┌───────────────────────┴──────┐",
            "│   FROZEN BASE WEIGHT W_0    │           │  TRAINABLE LoRA ADAPTER B·A  │",
            "│ • Shape: [d_out x d_in]     │           │ • Matrix B: [d_out x r=16]   │",
            "│ • 16-bit bf16 OR 4-bit NF4  │           │   (Initialized to zeros: 0)  │",
            "│ • requires_grad = False     │           │ • Matrix A: [r=16 x d_in]    │",
            "│ • 0 optimizer states        │           │   (Initialized ~ N(0, sigma²))│",
            "│ • Preserves pretrained      │           │ • Always in 16-bit bfloat16  │",
            "│   German language knowledge │           │ • Updated by AdamW optimizer │",
            "└──────▲──────────────────────┘           └───────────────▲──────────────┘",
            "       └──────────────────────────┐     ┌─────────────────┘",
            "                                   ╲   ╱",
            "                           Input Activation x (bfloat16)",
        ],
        accent_color=TEAL,
        font_size=10.0,
    )

    add_card(
        s5,
        left=7.1,
        top=1.4,
        width=5.6,
        height=5.6,
        title="How LoRA Works Mathematically & Architecturally",
        bullets=[
            ("The Forward Pass Equation:", "h = W_0 x + (α / r) · B · A · x, where r = 16 is the rank and α = 32 is the scaling factor (α/r = 2.0)."),
            ("Safe Zero-Initialization:", "Matrix A is sampled from a Gaussian; Matrix B is initialized to 0. At step 0, B·A = 0, so the model behaves identically to the pretrained base model."),
            ("All 7 Linear Projections Modified per Block:", "In each of the 26 Gemma-2 transformer layers, we attach LoRA to Attention (q_proj, k_proj, v_proj, o_proj) and GeGLU MLP (gate_proj, up_proj, down_proj)."),
            ("99.21% Parameter Reduction:", "Frozen Base Parameters: 2,614,341,888 (99.21%) vs. Trainable LoRA Parameters: 20,766,720 (0.79% = 798,720 per layer × 26 layers)."),
            ("Tiny Disk Checkpoints:", "Only the 20.8M adapter parameters (~41 MB in bf16) are saved to models/adapters/, making version control and deployment effortless."),
        ],
        bg_color=BG_TEAL,
        accent_color=TEAL,
    )

    # =========================================================================
    # SLIDE 6: CONCEPT 3 — 4-BIT NF4 QUANTIZATION (QLoRA)
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(
        s6,
        "5. Core Concept III — 4-bit NormalFloat Quantization (QLoRA)",
        "How Dettmers et al. (2023) compress frozen base weights by 72% while keeping gradients in 16-bit precision",
        6,
        total_slides,
    )

    add_card(
        s6,
        left=0.6,
        top=1.4,
        width=3.9,
        height=5.6,
        title="1. 4-Bit NormalFloat (NF4)",
        bullets=[
            ("Information-Theoretic Optimal:", "Pretrained neural network weights follow a zero-centered normal (Gaussian) distribution."),
            ("Equal-Mass Quantiles:", "NF4 places its 16 quantization bins (2^4 = 16 levels) so each bin covers an equal area under N(0, 1)."),
            ("Higher Fidelity Near Zero:", "Concentrates resolution around 0 where most weights live, outperforming uniform 4-bit integer (FP4/INT4)."),
            ("Blockwise Scaling:", "Weights are normalized in blocks of 64 values so outlier weights don't distort the entire tensor."),
        ],
        bg_color=BG_AMBER,
        accent_color=AMBER,
    )

    add_card(
        s6,
        left=4.7,
        top=1.4,
        width=3.9,
        height=5.6,
        title="2. Double Quantization (DQ)",
        bullets=[
            ("Quantizing the Scales:", "Every block of 64 weights needs a scaling constant c_1. In 32-bit float, those constants add 0.5 bits per parameter."),
            ("Second-Stage Compression:", "Double quantization (bnb_4bit_use_double_quant=True) quantizes the scaling constants themselves into 8-bit floats (FP8) in blocks of 256."),
            ("Memory Saved:", "Reduces scaling overhead from 0.5 bits/param down to 0.127 bits/param (~0.37 bits saved per weight across 2.6B params)."),
        ],
        bg_color=BG_LIGHT,
        accent_color=BLUE,
    )

    add_card(
        s6,
        left=8.8,
        top=1.4,
        width=3.9,
        height=5.6,
        title="3. Storage vs. Compute Dtype",
        bullets=[
            ("Storage Dtype = 4-bit NF4:", "The 2.61B frozen base weights W_0 sit in GPU VRAM in 4-bit NF4 format (~1.4 GB instead of ~5.0 GB)."),
            ("Compute Dtype = 16-bit bf16:", "During forward & backward passes, W_0 blocks are dequantized on-the-fly to 16-bit bfloat16 for matrix multiplication."),
            ("16-bit LoRA Gradients:", "LoRA adapters (A, B) are NEVER quantized—they stay in 16-bit bfloat16 so gradient updates lose zero precision."),
        ],
        bg_color=BG_TEAL,
        accent_color=TEAL,
    )

    # =========================================================================
    # SLIDE 7: 16-BIT LoRA vs. 4-BIT QLoRA COMPARISON TABLE
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(
        s7,
        "6. Head-to-Head: 16-bit bfloat16 LoRA vs. 4-bit NF4 QLoRA",
        "Architectural, hardware, and performance comparison on google/gemma-2-2b-it (GermEval 2017 Task C)",
        7,
        total_slides,
    )

    # Add a structured table
    rows, cols = 8, 4
    table_shape = s7.shapes.add_table(rows, cols, Inches(0.6), Inches(1.45), Inches(12.133), Inches(4.3))
    table = table_shape.table
    table.columns[0].width = Inches(3.133)
    table.columns[1].width = Inches(2.8)
    table.columns[2].width = Inches(3.1)
    table.columns[3].width = Inches(3.1)

    headers = [
        "Dimension / Metric",
        "Full Fine-Tuning (Reference)",
        "16-bit LoRA (configs/lora.yaml)",
        "4-bit QLoRA (configs/qlora.yaml)",
    ]
    data_rows = [
        ["Frozen Base Weights (W_0)", "None (All 2.61B updated)", "16-bit bfloat16 (~5.0 GB VRAM)", "4-bit NF4 + Double Quant (~1.4 GB)"],
        ["Trainable Parameters", "2,614,341,888 (100%)", "20,766,720 (0.79%, r=16, α=32)", "20,766,720 (0.79%, r=16, α=32)"],
        ["Adapter & Compute Dtype", "16-bit bfloat16", "16-bit bfloat16", "16-bit bfloat16 (on-the-fly dequant)"],
        ["Peak Training VRAM (Gemma-2-2B)", "> 32.0 GB (OOM on L4/T4)", "7.2 GB (Fits L4 24GB & T4 15GB)", "2.9 GB (Fits even 8GB consumer GPU)"],
        ["Step Latency Overhead", "Baseline (1.0x)", "Fastest PEFT (No dequantization)", "~15–20% slower (NF4 dequant kernel)"],
        ["Task C2 Strict Micro-F1 (test_syn)", "N/A", "0.7204 (95% CI: [0.706, 0.735])", "0.7191 (Δ = -0.0013, indistinguishable)"],
        ["JSON Schema Validity (test_syn)", "N/A", "100.0% VALID_EXACT", "100.0% VALID_EXACT"],
    ]

    for c_idx, h_text in enumerate(headers):
        cell = table.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.text = h_text
        p.font.bold = True
        p.font.size = Pt(12.5)
        p.font.color.rgb = WHITE
        p.font.name = "Calibri"

    for r_idx, row_vals in enumerate(data_rows, start=1):
        bg = WHITE if r_idx % 2 == 1 else BG_LIGHT
        for c_idx, val in enumerate(row_vals):
            cell = table.cell(r_idx, c_idx)
            cell.fill.solid()
            if c_idx == 2:
                cell.fill.fore_color.rgb = BG_BLUE if r_idx % 2 == 0 else WHITE
            elif c_idx == 3:
                cell.fill.fore_color.rgb = BG_TEAL if r_idx % 2 == 0 else WHITE
            else:
                cell.fill.fore_color.rgb = bg
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(11.5)
            p.font.bold = c_idx == 0 or r_idx in (4, 6)
            p.font.color.rgb = NAVY if c_idx == 0 else SLATE_DARK
            p.font.name = "Calibri"

    add_card(
        s7,
        left=0.6,
        top=5.95,
        width=12.133,
        height=1.1,
        title="Practical Takeaway for Google Colab (Notebook 11)",
        bullets=[
            ("Single-Flag Toggle in Step 3:", "Set USE_4BIT_QUANTIZATION = True (default) for 4-bit NF4 QLoRA (~2.9 GB VRAM), or USE_4BIT_QUANTIZATION = False for 16-bit bfloat16 LoRA (~7.2 GB VRAM). Both achieve ~0.720 Micro-F1 on GermEval 2017 Task C2!"),
        ],
        bg_color=BG_AMBER,
        accent_color=AMBER,
        title_size=14,
        body_size=12,
    )

    # =========================================================================
    # SLIDE 8: EMPIRICAL BENCHMARK SUMMARY & TAKEAWAYS
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(
        s8,
        "7. Empirical Results & Key Takeaways on GermEval 2017",
        "Summary across Prompting Baselines, 16-bit LoRA, and 4-bit QLoRA on Synchronic (test_syn) & Diachronic (test_dia) Splits",
        8,
        total_slides,
    )

    # Benchmark table on left
    rows8, cols8 = 7, 5
    t8_shape = s8.shapes.add_table(rows8, cols8, Inches(0.6), Inches(1.45), Inches(7.3), Inches(3.8))
    t8 = t8_shape.table
    t8.columns[0].width = Inches(2.5)
    t8.columns[1].width = Inches(1.2)
    t8.columns[2].width = Inches(1.2)
    t8.columns[3].width = Inches(1.2)
    t8.columns[4].width = Inches(1.2)

    h8 = ["Method / Run ID", "Valid JSON", "C1 Micro-F1", "C2 Micro-F1", "Dia C2 F1"]
    d8 = [
        ["P0 Minimal Zero-Shot", "91.4%", "0.312", "0.214", "0.201"],
        ["P1 Guideline Zero-Shot", "95.8%", "0.368", "0.258", "0.244"],
        ["P2 Few-Shot (2-Shot)", "98.1%", "0.491", "0.371", "0.355"],
        ["P3 Optimized 10-Shot", "99.4%", "0.569", "0.453", "0.438"],
        ["L1 16-bit LoRA (r=16)", "100.0%", "0.781", "0.720", "0.698"],
        ["Q1 4-bit QLoRA (r=16)", "100.0%", "0.779", "0.719", "0.696"],
    ]

    for c_i, text in enumerate(h8):
        c = t8.cell(0, c_i)
        c.fill.solid()
        c.fill.fore_color.rgb = NAVY
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = c.text_frame.paragraphs[0]
        p.text = text
        p.font.bold = True
        p.font.size = Pt(11.5)
        p.font.color.rgb = WHITE
        p.font.name = "Calibri"

    for r_i, r_vals in enumerate(d8, start=1):
        is_ft = r_i >= 5
        for c_i, val in enumerate(r_vals):
            c = t8.cell(r_i, c_i)
            c.fill.solid()
            c.fill.fore_color.rgb = BG_TEAL if is_ft else (WHITE if r_i % 2 == 1 else BG_LIGHT)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = c.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(11.5)
            p.font.bold = is_ft or c_i == 0
            p.font.color.rgb = TEAL if (is_ft and c_i >= 2) else NAVY
            p.font.name = "Calibri"

    add_card(
        s8,
        left=8.1,
        top=1.45,
        width=4.6,
        height=3.8,
        title="Sample Efficiency (Q2 Learning Curve)",
        bullets=[
            ("N = 250 docs (1.3%):", "0.512 C2 Micro-F1 — already beats 10-shot prompting (0.453)!"),
            ("N = 1,000 docs (5.1%):", "0.609 C2 Micro-F1 — trains in <5 minutes on Colab L4."),
            ("N = 5,000 docs (25.7%):", "0.684 C2 Micro-F1 — captures 95% of full-data performance."),
            ("N = 19,432 docs (100%):", "0.719 C2 Micro-F1 — full convergence across all 19 categories."),
        ],
        bg_color=BG_BLUE,
        accent_color=BLUE,
    )

    add_card(
        s8,
        left=0.6,
        top=5.45,
        width=12.1,
        height=1.6,
        title="Summary of Core Concepts",
        bullets=[
            ("1. Data & Loss Masking:", "Deterministic JSON targets + Completion-Only Loss (labels = -100 on prompt tokens) focus 100% of gradient updates on structured prediction."),
            ("2. Low-Rank Adapters (LoRA):", "Training just 0.79% of parameters (r=16 across all 7 attention/MLP projections) unlocks +26.7 Micro-F1 over 10-shot prompting."),
            ("3. 4-Bit NF4 Quantization (QLoRA):", "Cuts peak training VRAM from 7.2 GB to 2.9 GB with virtually zero accuracy loss (0.7191 vs. 0.7204 F1)."),
        ],
        bg_color=BG_LIGHT,
        accent_color=NAVY,
        title_size=15,
        body_size=12,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))
    return out


if __name__ == "__main__":
    saved = build_presentation()
    print(f"Presentation saved to: {saved} ({saved.stat().st_size / 1024:.1f} KB)")
