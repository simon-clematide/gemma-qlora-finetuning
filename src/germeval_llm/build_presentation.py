"""Generate two clean, concise, low-density PowerPoint presentations (.pptx):
1. docs/gemma_qlora_core_concepts.pptx — Technical overview of Data, LoRA Adapters, and 4-bit Quantization.
2. docs/gemma_qlora_task_and_results.pptx — Focused deck explaining the GermEval 2017 Dataset, Tasks (C & D),
   Approaches, and Empirical Results / Takeaways (with embedded charts).
"""

from __future__ import annotations

from pathlib import Path
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


# Color Palette (Modern Academic Navy / Slate / Teal / Amber)
NAVY = RGBColor(15, 23, 42)           # #0F172A - Headers / Dark accents
BLUE = RGBColor(37, 99, 235)          # #2563EB - Primary accent
TEAL = RGBColor(13, 148, 136)         # #0D9488 - Secondary / LoRA trainable accent
AMBER = RGBColor(217, 119, 6)         # #D97706 - Quantization / Highlight accent
SLATE_DARK = RGBColor(51, 65, 85)     # #334155 - Body text
SLATE_MUTED = RGBColor(100, 116, 139) # #64748B - Subtitles / Captions
BG_LIGHT = RGBColor(248, 250, 252)    # #F8FAFC - Card background
BG_BLUE = RGBColor(239, 246, 255)     # #EFF6FF - Soft blue card
BG_TEAL = RGBColor(240, 253, 250)     # #F0FDFA - Soft teal card
BG_AMBER = RGBColor(255, 251, 235)    # #FFFBEB - Soft amber card
BORDER_GRAY = RGBColor(203, 213, 225) # #CBD5E1 - Card border
WHITE = RGBColor(255, 255, 255)


def add_header(slide, title_text: str, subtitle_text: str, slide_num: int, total_slides: int) -> None:
    """Add a clean, spacious top banner with title, subtitle, and slide number."""
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.12))
    bar.fill.solid()
    bar.fill.fore_color.rgb = BLUE
    bar.line.fill.background()

    tb = slide.shapes.add_textbox(Inches(0.7), Inches(0.28), Inches(11.0), Inches(0.95))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p = tf.paragraphs[0]
    p.text = title_text
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = NAVY
    p.font.name = "Calibri"

    p2 = tf.add_paragraph()
    p2.text = subtitle_text
    p2.font.size = Pt(15)
    p2.font.color.rgb = SLATE_MUTED
    p2.font.name = "Calibri"

    num_box = slide.shapes.add_textbox(Inches(11.7), Inches(0.32), Inches(0.95), Inches(0.4))
    ntf = num_box.text_frame
    np = ntf.paragraphs[0]
    np.text = f"{slide_num} / {total_slides}"
    np.alignment = PP_ALIGN.RIGHT
    np.font.size = Pt(13)
    np.font.bold = True
    np.font.color.rgb = SLATE_MUTED
    np.font.name = "Calibri"

    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.7), Inches(1.28), Inches(11.933), Inches(0.02))
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
    title_size: int = 19,
    body_size: int = 15,
) -> None:
    """Add a spacious card with short, punchy bullets."""
    card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    card.adjustments[0] = 0.025
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    card.line.color.rgb = BORDER_GRAY
    card.line.width = Pt(1)

    strip = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(left), Inches(top + 0.1), Inches(0.09), Inches(height - 0.2)
    )
    strip.fill.solid()
    strip.fill.fore_color.rgb = accent_color
    strip.line.fill.background()

    tb = slide.shapes.add_textbox(
        Inches(left + 0.28), Inches(top + 0.2), Inches(width - 0.45), Inches(height - 0.35)
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
    p_title.space_after = Pt(10)

    for item in bullets:
        p = tf.add_paragraph()
        p.space_after = Pt(8)
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
    font_size: int = 12,
) -> None:
    """Add a clean dark-themed monospace box."""
    box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    box.adjustments[0] = 0.025
    box.fill.solid()
    box.fill.fore_color.rgb = NAVY
    box.line.color.rgb = accent_color
    box.line.width = Pt(1.5)

    tb = slide.shapes.add_textbox(
        Inches(left + 0.24), Inches(top + 0.18), Inches(width - 0.48), Inches(height - 0.32)
    )
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p0 = tf.paragraphs[0]
    p0.text = title
    p0.font.size = Pt(14)
    p0.font.bold = True
    p0.font.color.rgb = RGBColor(96, 165, 250)
    p0.font.name = "Calibri"
    p0.space_after = Pt(8)

    for line in code_lines:
        p = tf.add_paragraph()
        p.text = line
        p.font.size = Pt(font_size)
        p.font.color.rgb = RGBColor(226, 232, 240)
        p.font.name = "Courier New"
        p.space_after = Pt(2)


def build_core_concepts_presentation(
    output_path: str | Path = "docs/gemma_qlora_core_concepts.pptx",
) -> Path:
    """Build Deck 1: Core Concepts (Data, LoRA Adapters, 4-bit Quantization) with exact benchmark numbers."""
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]
    total_slides = 8

    # SLIDE 1: TITLE
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = NAVY
    bg1.line.fill.background()

    stripe1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.4), Inches(0.12), Inches(4.7))
    stripe1.fill.solid()
    stripe1.fill.fore_color.rgb = TEAL
    stripe1.line.fill.background()

    tb1 = s1.shapes.add_textbox(Inches(1.2), Inches(1.4), Inches(11.2), Inches(4.7))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p = tf1.paragraphs[0]
    p.text = "PARAMETER-EFFICIENT LLM ADAPTATION"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = RGBColor(45, 212, 191)
    p.font.name = "Calibri"
    p.space_after = Pt(12)

    p2 = tf1.add_paragraph()
    p2.text = "Fine-Tuning Gemma-2-2B-IT with LoRA & QLoRA"
    p2.font.size = Pt(38)
    p2.font.bold = True
    p2.font.color.rgb = WHITE
    p2.font.name = "Calibri"
    p2.space_after = Pt(10)

    p3 = tf1.add_paragraph()
    p3.text = "Core Concepts: Instruction Data, Low-Rank Adapters, and 4-bit Quantization"
    p3.font.size = Pt(20)
    p3.font.color.rgb = RGBColor(148, 163, 184)
    p3.font.name = "Calibri"

    pillars = [
        ("1. DATA & LOSS MASKING", "GermEval 2017 Reviews → JSON\nCompletion-Only Loss (-100)", BLUE),
        ("2. LOW-RANK ADAPTERS (LoRA)", "Freeze 99.2% Base Weights\nTrain Only 0.8% Rank-16 Weights", TEAL),
        ("3. 4-BIT QUANTIZATION (QLoRA)", "72% Base Weight Compression\nTrains in 2.9 GB VRAM on Colab", AMBER),
    ]
    for idx, (ptitle, pdesc, pcol) in enumerate(pillars):
        x = 1.2 + idx * 3.8
        box = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(4.4), Inches(3.55), Inches(1.65))
        box.adjustments[0] = 0.04
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(30, 41, 59)
        box.line.color.rgb = pcol
        box.line.width = Pt(1.5)

        btf = box.text_frame
        btf.word_wrap = True
        btf.margin_left = btf.margin_right = Inches(0.2)
        btf.margin_top = Inches(0.2)

        bp1 = btf.paragraphs[0]
        bp1.text = ptitle
        bp1.font.size = Pt(13)
        bp1.font.bold = True
        bp1.font.color.rgb = pcol
        bp1.font.name = "Calibri"
        bp1.space_after = Pt(8)

        bp2 = btf.add_paragraph()
        bp2.text = pdesc
        bp2.font.size = Pt(14)
        bp2.font.color.rgb = WHITE
        bp2.font.name = "Calibri"

    # SLIDE 2: MOTIVATION
    s2 = prs.slides.add_slide(blank_layout)
    add_header(
        s2,
        "1. Motivation: Why Parameter-Efficient Fine-Tuning?",
        "Comparing Prompting, Full Fine-Tuning, and LoRA/QLoRA on Gemma-2-2B-IT",
        2,
        total_slides,
    )

    add_card(
        s2,
        left=0.7,
        top=1.6,
        width=3.8,
        height=5.2,
        title="In-Context Prompting",
        bullets=[
            ("Hard Domain Rules:", "19 German aspect categories have subtle annotation boundaries."),
            ("Accuracy Ceiling:", "Zero-shot reaches 0.587 C2 F1; even 10-shot plateaus at 0.703 F1."),
            ("High Token Cost:", "10-shot prepends ~1,760 tokens per document (2.7x more tokens)."),
        ],
        bg_color=BG_LIGHT,
        accent_color=SLATE_DARK,
    )

    add_card(
        s2,
        left=4.75,
        top=1.6,
        width=3.8,
        height=5.2,
        title="Full Fine-Tuning",
        bullets=[
            (">32 GB VRAM Needed:", "Updating all 2.61B weights + AdamW states overflows consumer GPUs."),
            ("Catastrophic Forgetting:", "Overwriting all weights risks degrading general German fluency."),
            ("5 GB Checkpoints:", "Every saved experiment copies the entire 2.61B model."),
        ],
        bg_color=BG_AMBER,
        accent_color=AMBER,
    )

    add_card(
        s2,
        left=8.8,
        top=1.6,
        width=3.8,
        height=5.2,
        title="LoRA & QLoRA (Ours)",
        bullets=[
            ("+14.5 F1 over 10-Shot:", "Jumps from 0.587 (0-shot) & 0.703 (10-shot) to 0.849 Micro-F1."),
            ("100% Valid JSON:", "Learns the exact output schema without post-processing hacks."),
            ("Tiny Memory & Disk:", "Trains in 2.9 GB VRAM (QLoRA); saves 41 MB adapter files."),
        ],
        bg_color=BG_TEAL,
        accent_color=TEAL,
    )

    # SLIDE 3: CONCEPT I (PART A)
    s3 = prs.slides.add_slide(blank_layout)
    add_header(
        s3,
        "2. Core Concept I — Data: GermEval 2017 to Structured JSON",
        "26,209 Deutsche Bahn customer reviews converted into deterministic JSON targets",
        3,
        total_slides,
    )

    add_card(
        s3,
        left=0.7,
        top=1.6,
        width=5.6,
        height=5.2,
        title="Task C: Aspect–Polarity Classification",
        bullets=[
            ("Dataset Splits:", "19,432 Train | 2,369 Dev | 2,566 Test (Syn) | 1,842 Test (Dia)."),
            ("Label Space:", "19 aspect categories × 3 polarities (positive, negative, neutral)."),
            ("Out-of-Domain Posts:", "relevance=false (21% of docs) maps cleanly to {\"labels\": []}."),
            ("Deterministic Order:", "Labels are deduplicated and sorted alphabetically for stable training."),
        ],
        bg_color=BG_BLUE,
        accent_color=BLUE,
    )

    add_code_box(
        s3,
        left=6.6,
        top=1.6,
        width=6.0,
        height=2.45,
        title="Input: Raw Customer Review (XML)",
        code_lines=[
            '"@DB_Bahn Mal wieder 45 Minuten Verspätung',
            ' im ICE nach München und das WLAN geht',
            ' auch nicht!"',
            "",
            "Gold Annotations:",
            " • Zugfahrt#Pünktlichkeit -> negative",
            " • Sonstige_Unregelmässigkeiten -> negative",
        ],
        accent_color=BLUE,
        font_size=12,
    )

    add_code_box(
        s3,
        left=6.6,
        top=4.35,
        width=6.0,
        height=2.45,
        title="Output: Deterministic SFT Target (JSON)",
        code_lines=[
            "{",
            '  "labels": [',
            '    "Sonstige_Unregelmässigkeiten:negative",',
            '    "Zugfahrt:negative"',
            "  ]",
            "}",
        ],
        accent_color=TEAL,
        font_size=13,
    )

    # SLIDE 4: CONCEPT I (PART B)
    s4 = prs.slides.add_slide(blank_layout)
    add_header(
        s4,
        "3. Core Concept I — Data: Completion-Only Loss Masking",
        "Focusing 100% of gradient updates on the JSON answer via labels = -100",
        4,
        total_slides,
    )

    add_code_box(
        s4,
        left=0.7,
        top=1.6,
        width=11.9,
        height=2.3,
        title="Token Sequence Inside trl.SFTTrainer (completion_only_loss=True)",
        code_lines=[
            "Tokens : <bos> <start_of_turn>user ... [~660 Prompt Tokens] ... <start_of_turn>model\\n {\"labels\": [\"Zugfahrt:negative\"]} <end_of_turn>",
            "Labels : -100       -100                      -100                   -100        -100   \"{\"  \"labels\" ... \"Zugfahrt:negative\" \"]}\" <end_of_turn>",
            "         └──────────────────────────────────────┬────────────────────────────────────┘  └──────────────────────┬──────────────────────┘",
            "                            IGNORED BY LOSS (PyTorch ignore_index = -100)                          ACTIVE CROSS-ENTROPY LOSS (~20 Tokens)",
        ],
        accent_color=TEAL,
        font_size=11.5,
    )

    add_card(
        s4,
        left=0.7,
        top=4.2,
        width=5.75,
        height=2.6,
        title="The 35:1 Token Asymmetry Problem",
        bullets=[
            ("Long Prompt (~660 tokens):", "Contains static task rules, 19 categories, and review text."),
            ("Short Answer (~20 tokens):", "Contains only the compact JSON classification."),
        ],
        bg_color=BG_LIGHT,
        accent_color=BLUE,
    )

    add_card(
        s4,
        left=6.85,
        top=4.2,
        width=5.75,
        height=2.6,
        title="Why Masking Prompt Tokens Works",
        bullets=[
            ("Zero Wasted Capacity:", "Adapter never wastes gradients memorizing the German prompt."),
            ("Fast Convergence:", "100% of backpropagation directly optimizes JSON label prediction."),
        ],
        bg_color=BG_TEAL,
        accent_color=TEAL,
    )

    # SLIDE 5: CONCEPT II — LoRA
    s5 = prs.slides.add_slide(blank_layout)
    add_header(
        s5,
        "4. Core Concept II — Low-Rank Adaptation (LoRA)",
        "Freezing 2.61B base weights and training rank-16 bypass matrices (A, B)",
        5,
        total_slides,
    )

    add_code_box(
        s5,
        left=0.7,
        top=1.6,
        width=6.1,
        height=5.2,
        title="LoRA Bypass Inside Every Linear Layer",
        code_lines=[
            "        Output: h = W_0·x + (alpha / r) · B·(A·x)",
            "                            ▲",
            "                           (+)  ◄── Sum in bfloat16",
            "                          ╱   ╲",
            "      ┌──────────────────┘     └──────────────────┐",
            "┌─────┴───────────────────┐   ┌───────────────────┴─────┐",
            "│  FROZEN BASE WEIGHT W_0 │   │  TRAINABLE LoRA (B · A) │",
            "│ • [d_out x d_in]        │   │ • B: [d_out x 16] (= 0) │",
            "│ • 16-bit OR 4-bit NF4   │   │ • A: [16 x d_in]  (~ N) │",
            "│ • requires_grad = False │   │ • requires_grad = True  │",
            "│ • 0 optimizer states    │   │ • Updated by AdamW      │",
            "└─────▲───────────────────┘   └───────────────────▲─────┘",
            "      └──────────────────┐     ┌──────────────────┘",
            "                          ╲   ╱",
            "                   Input Activation x",
        ],
        accent_color=TEAL,
        font_size=11.0,
    )

    add_card(
        s5,
        left=7.1,
        top=1.6,
        width=5.5,
        height=5.2,
        title="Key Properties of Our LoRA Setup",
        bullets=[
            ("Low-Rank Bottleneck:", "Rank r = 16, scaling α = 32 (α/r = 2.0)."),
            ("Safe Initialization:", "B = 0 at step 0, so training starts strictly from the exact base model."),
            ("All 7 Projections Adapted:", "Applied to Attention (q, k, v, o) and MLP (gate, up, down) across all 26 layers."),
            ("99.21% Parameter Savings:", "Only 20.8M trainable weights vs. 2.61B frozen base weights."),
        ],
        bg_color=BG_TEAL,
        accent_color=TEAL,
    )

    # SLIDE 6: CONCEPT III — QLoRA
    s6 = prs.slides.add_slide(blank_layout)
    add_header(
        s6,
        "5. Core Concept III — 4-bit Quantization (QLoRA)",
        "Compressing frozen base weights by 72% while keeping gradients in 16-bit",
        6,
        total_slides,
    )

    add_card(
        s6,
        left=0.7,
        top=1.6,
        width=3.8,
        height=5.2,
        title="1. NormalFloat4 (NF4)",
        bullets=[
            ("Bell-Curve Optimal:", "Pretrained weights follow a zero-centered Gaussian N(0, σ²)."),
            ("16 Quantile Bins:", "Places more 4-bit bins near 0 where most weights are concentrated."),
            ("Blockwise Scaling:", "Normalizes weights in blocks of 64 to isolate outliers."),
        ],
        bg_color=BG_AMBER,
        accent_color=AMBER,
    )

    add_card(
        s6,
        left=4.75,
        top=1.6,
        width=3.8,
        height=5.2,
        title="2. Double Quantization",
        bullets=[
            ("Compressing Scales:", "Each block of 64 weights stores a scaling factor."),
            ("FP32 → FP8 Scales:", "Quantizes the scaling factors themselves into 8-bit floats."),
            ("Extra VRAM Saved:", "Cuts scale overhead from 0.5 to 0.13 bits per parameter."),
        ],
        bg_color=BG_LIGHT,
        accent_color=BLUE,
    )

    add_card(
        s6,
        left=8.8,
        top=1.6,
        width=3.8,
        height=5.2,
        title="3. Mixed-Precision Flow",
        bullets=[
            ("4-Bit Storage (1.4 GB):", "Frozen base weights W_0 stay compressed in GPU RAM."),
            ("16-Bit Compute (bf16):", "W_0 is dequantized on-the-fly during matrix multiplication."),
            ("16-Bit LoRA Updates:", "Adapters A & B are never quantized—zero gradient loss!"),
        ],
        bg_color=BG_TEAL,
        accent_color=TEAL,
    )

    # SLIDE 7: 16-BIT LoRA vs. 4-BIT QLoRA
    s7 = prs.slides.add_slide(blank_layout)
    add_header(
        s7,
        "6. Comparison: 16-bit LoRA vs. 4-bit QLoRA",
        "Hardware requirements and accuracy on Gemma-2-2B-IT (GermEval 2017 Task C)",
        7,
        total_slides,
    )

    rows, cols = 6, 4
    table_shape = s7.shapes.add_table(rows, cols, Inches(0.7), Inches(1.6), Inches(11.933), Inches(3.8))
    table = table_shape.table
    table.columns[0].width = Inches(3.233)
    table.columns[1].width = Inches(2.7)
    table.columns[2].width = Inches(3.0)
    table.columns[3].width = Inches(3.0)

    headers = [
        "Feature",
        "Full Fine-Tuning",
        "16-bit LoRA (lora.yaml)",
        "4-bit QLoRA (qlora.yaml)",
    ]
    data_rows = [
        ["Frozen Base Weights (W_0)", "None (All updated)", "16-bit bfloat16 (5.0 GB)", "4-bit NF4 (1.4 GB, -72%)"],
        ["Trainable Parameters", "2,614M (100%)", "20.8M (0.79%, r=16)", "20.8M (0.79%, r=16)"],
        ["Peak Training VRAM", "> 32.0 GB", "7.2 GB (Fits Colab L4/T4)", "2.9 GB (Fits 8 GB GPU)"],
        ["Training Speed per Step", "Baseline", "Fastest (No dequant)", "~15% slower (NF4 dequant)"],
        ["Task C2 Micro-F1 (test_syn)", "N/A", "0.8486 Strict / 0.7370 Java", "0.8360 Strict / 0.7182 Java"],
    ]

    for c_idx, h_text in enumerate(headers):
        cell = table.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.text = h_text
        p.font.bold = True
        p.font.size = Pt(14)
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
            p.font.size = Pt(13)
            p.font.bold = c_idx == 0 or r_idx in (3, 5)
            p.font.color.rgb = NAVY if c_idx == 0 else SLATE_DARK
            p.font.name = "Calibri"

    add_card(
        s7,
        left=0.7,
        top=5.65,
        width=11.933,
        height=1.2,
        title="One-Line Toggle in Colab Demo (Notebook 11, Step 3)",
        bullets=[
            ("USE_4BIT_QUANTIZATION = True / False:", "Switch seamlessly between 4-bit QLoRA (2.9 GB VRAM) and 16-bit LoRA (7.2 GB VRAM) with <1.3% F1 difference."),
        ],
        bg_color=BG_AMBER,
        accent_color=AMBER,
        title_size=15,
        body_size=13.5,
    )

    # SLIDE 8: EMPIRICAL RESULTS SUMMARY
    s8 = prs.slides.add_slide(blank_layout)
    add_header(
        s8,
        "7. Empirical Results & Key Takeaways",
        "Benchmark results on GermEval 2017 (test_syn) and data efficiency",
        8,
        total_slides,
    )

    rows8, cols8 = 6, 4
    t8_shape = s8.shapes.add_table(rows8, cols8, Inches(0.7), Inches(1.6), Inches(6.8), Inches(3.6))
    t8 = t8_shape.table
    t8.columns[0].width = Inches(2.9)
    t8.columns[1].width = Inches(1.3)
    t8.columns[2].width = Inches(1.3)
    t8.columns[3].width = Inches(1.3)

    h8 = ["Method (test_syn)", "Valid JSON", "C1 Strict F1", "C2 Strict F1"]
    d8 = [
        ["P0 Minimal Zero-Shot", "98.1%", "0.555", "0.464"],
        ["P1 Guideline Zero-Shot", "99.7%", "0.668", "0.587"],
        ["P3 Optimized 10-Shot", "100.0%", "0.767", "0.703"],
        ["L1 16-bit LoRA (r=16)", "100.0%", "0.882", "0.849"],
        ["Q1 4-bit QLoRA (r=16)", "100.0%", "0.873", "0.836"],
    ]

    for c_i, text in enumerate(h8):
        c = t8.cell(0, c_i)
        c.fill.solid()
        c.fill.fore_color.rgb = NAVY
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = c.text_frame.paragraphs[0]
        p.text = text
        p.font.bold = True
        p.font.size = Pt(13)
        p.font.color.rgb = WHITE
        p.font.name = "Calibri"

    for r_i, r_vals in enumerate(d8, start=1):
        is_ft = r_i >= 4
        for c_i, val in enumerate(r_vals):
            c = t8.cell(r_i, c_i)
            c.fill.solid()
            c.fill.fore_color.rgb = BG_TEAL if is_ft else (WHITE if r_i % 2 == 1 else BG_LIGHT)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = c.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(13)
            p.font.bold = is_ft or c_i == 0
            p.font.color.rgb = TEAL if (is_ft and c_i >= 2) else NAVY
            p.font.name = "Calibri"

    add_card(
        s8,
        left=7.8,
        top=1.6,
        width=4.8,
        height=3.6,
        title="Sample Efficiency (QLoRA)",
        bullets=[
            ("250 docs (1.3%):", "0.648 F1 (beats 2-shot prompting)"),
            ("1,000 docs (5.1%):", "0.701 F1 (matches 10-shot in 1.6 min)"),
            ("2,000 docs (10.3%):", "0.722 F1 (surpasses 10-shot)"),
            ("19,432 docs (100%):", "0.836 F1 (+13.3 F1 over 10-shot)"),
        ],
        bg_color=BG_BLUE,
        accent_color=BLUE,
    )

    add_card(
        s8,
        left=0.7,
        top=5.45,
        width=11.9,
        height=1.45,
        title="Takeaway",
        bullets=[
            ("Data + Adapters + Quantization:", "Completion-masked JSON SFT + rank-16 LoRA + 4-bit NF4 quantization delivers +14.5 F1 over 10-shot prompting in under 3 GB of GPU memory."),
        ],
        bg_color=BG_LIGHT,
        accent_color=NAVY,
        title_size=16,
        body_size=14,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))
    return out


def build_task_and_results_presentation(
    output_path: str | Path = "docs/gemma_qlora_task_and_results.pptx",
) -> Path:
    """Build Deck 2: Focused presentation on the GermEval 2017 Data, Task, Approaches, and Main Takeaways."""
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]
    total_slides = 7

    # =========================================================================
    # SLIDE 1: TITLE & MAIN EXPERIMENT TAKEAWAY
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = NAVY
    bg1.line.fill.background()

    stripe1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.3), Inches(0.12), Inches(4.9))
    stripe1.fill.solid()
    stripe1.fill.fore_color.rgb = TEAL
    stripe1.line.fill.background()

    tb1 = s1.shapes.add_textbox(Inches(1.2), Inches(1.3), Inches(11.2), Inches(4.9))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p = tf1.paragraphs[0]
    p.text = "GERMEVAL 2017 ASPECT-BASED SENTIMENT ANALYSIS"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = RGBColor(45, 212, 191)
    p.font.name = "Calibri"
    p.space_after = Pt(12)

    p2 = tf1.add_paragraph()
    p2.text = "Prompting vs. Fine-Tuning on German Customer Feedback"
    p2.font.size = Pt(36)
    p2.font.bold = True
    p2.font.color.rgb = WHITE
    p2.font.name = "Calibri"
    p2.space_after = Pt(10)

    p3 = tf1.add_paragraph()
    p3.text = "Understanding the Dataset, the Tasks, and Why Fine-Tuning Beats Prompting by +14.5 F1"
    p3.font.size = Pt(19)
    p3.font.color.rgb = RGBColor(148, 163, 184)
    p3.font.name = "Calibri"

    kpis = [
        ("THE BENCHMARK", "26,209 German Reviews\n19 Aspect Categories × 3 Polarities", BLUE),
        ("THE MAIN RESULT", "Fine-Tuning: 0.849 Micro-F1\n10-Shot Prompting: 0.703 Micro-F1", TEAL),
        ("THE EFFICIENCY", "1,000 Training Examples (~2 min)\nAlready Match 10-Shot Prompting", AMBER),
    ]
    for idx, (ktitle, kdesc, kcol) in enumerate(kpis):
        x = 1.2 + idx * 3.8
        box = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(4.4), Inches(3.55), Inches(1.7))
        box.adjustments[0] = 0.04
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(30, 41, 59)
        box.line.color.rgb = kcol
        box.line.width = Pt(1.5)

        btf = box.text_frame
        btf.word_wrap = True
        btf.margin_left = btf.margin_right = Inches(0.2)
        btf.margin_top = Inches(0.2)

        bp1 = btf.paragraphs[0]
        bp1.text = ktitle
        bp1.font.size = Pt(13)
        bp1.font.bold = True
        bp1.font.color.rgb = kcol
        bp1.font.name = "Calibri"
        bp1.space_after = Pt(8)

        bp2 = btf.add_paragraph()
        bp2.text = kdesc
        bp2.font.size = Pt(14.5)
        bp2.font.color.rgb = WHITE
        bp2.font.name = "Calibri"

    # =========================================================================
    # SLIDE 2: THE DATA — GERMEVAL 2017 CORPUS & DISTRIBUTION
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(
        s2,
        "1. The Dataset: Real-World German Customer Feedback",
        "26,209 social media, news, and forum posts about Deutsche Bahn (GermEval 2017 Shared Task)",
        2,
        total_slides,
    )

    add_card(
        s2,
        left=0.7,
        top=1.55,
        width=5.4,
        height=5.35,
        title="Corpus Structure & Challenges",
        bullets=[
            ("4 Official Splits (26,209 Docs):", "Train (19,432) | Dev (2,369) | Synchronic Test (2,566) | Diachronic Test (1,842, collected months later)."),
            ("Noisy User-Generated Text:", "Tweets, forum threads, and news comments with irony, slang, hashtags, and typos."),
            ("21% Out-of-Domain Noise:", "Over a fifth of posts mention trains/stations only in passing (relevance=false) and carry 0 opinions."),
            ("Severe Class Imbalance:", "Dominated by Allgemein, Zugfahrt, and Sonstige_Unregelmässigkeiten; rare categories appear in <1% of docs."),
        ],
        bg_color=BG_BLUE,
        accent_color=BLUE,
        title_size=18,
        body_size=14,
    )

    cat_img = Path("results/figures/dataset_category_distribution.png")
    if cat_img.exists():
        s2.shapes.add_picture(str(cat_img), Inches(6.35), Inches(1.65), width=Inches(6.25))

    # =========================================================================
    # SLIDE 3: THE TASK — CONCRETE EXAMPLE (TASK C & TASK D)
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(
        s3,
        "2. The Task: From Unstructured Text to Structured Opinions",
        "What the model receives as input and what it must predict for Task C and Task D",
        3,
        total_slides,
    )

    add_code_box(
        s3,
        left=0.7,
        top=1.55,
        width=11.9,
        height=1.55,
        title="Example Customer Post (Input to Gemma-2-2B-IT)",
        code_lines=[
            '"@DB_Bahn Mal wieder 45 Minuten Verspätung im ICE nach München und das WLAN geht auch nicht! Personal war aber freundlich."',
            '("Once again 45 minutes delay on the ICE to Munich and the Wi-Fi isn\'t working either! Staff was friendly though.")',
        ],
        accent_color=BLUE,
        font_size=12.5,
    )

    add_card(
        s3,
        left=0.7,
        top=3.35,
        width=5.75,
        height=3.55,
        title="Task C: Aspect–Polarity Classification (Primary)",
        bullets=[
            ("Goal:", "Identify every (Category:polarity) pair expressed in the post."),
            ("19 Categories × 3 Polarities:", "e.g., Zugfahrt:negative, Sonstige_Unregelmässigkeiten:negative, Atmosphäre:positive."),
            ("Two Sub-Metrics:", "Task C1 = Category only | Task C2 = Category + Polarity."),
            ("Predicted JSON Output:", '{"labels": ["Atmosphäre:positive", "Sonstige_Unregelmässigkeiten:negative", "Zugfahrt:negative"]}'),
        ],
        bg_color=BG_TEAL,
        accent_color=TEAL,
        title_size=17,
        body_size=13.5,
    )

    add_card(
        s3,
        left=6.85,
        top=3.35,
        width=5.75,
        height=3.55,
        title="Task D: Opinion Target Extraction (Span-Level)",
        bullets=[
            ("Goal:", "Extract the exact text phrase (target word/span) that triggered each specific opinion."),
            ("Character-Grounded Evaluation:", "Evaluated by exact character offset match in the original German sentence."),
            ("Predicted JSON Output:", '{"targets": [{"category": "Zugfahrt", "polarity": "negative", "target": "Verspätung"}, {"category": "Sonstige_Unregelmässigkeiten", "polarity": "negative", "target": "WLAN"}]}'),
        ],
        bg_color=BG_LIGHT,
        accent_color=BLUE,
        title_size=17,
        body_size=13.5,
    )

    # =========================================================================
    # SLIDE 4: THE APPROACHES COMPARED
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(
        s4,
        "3. The Approaches Compared on Gemma-2-2B-IT",
        "All methods use the exact same 2.6B base model and output the exact same JSON schema",
        4,
        total_slides,
    )

    add_card(
        s4,
        left=0.7,
        top=1.6,
        width=3.8,
        height=5.2,
        title="1. Zero-Shot Prompting",
        bullets=[
            ("P0 (Minimal):", "Lists only the 19 category names and 3 polarities (~285 tokens/doc)."),
            ("P1 (Guidelines):", "Includes official German definitions for all 19 categories (~660 tokens/doc)."),
            ("No Training Needed:", "Model weights remain 100% untouched."),
        ],
        bg_color=BG_LIGHT,
        accent_color=SLATE_DARK,
    )

    add_card(
        s4,
        left=4.75,
        top=1.6,
        width=3.8,
        height=5.2,
        title="2. Few-Shot Prompting",
        bullets=[
            ("P2 (2, 5, 10-Shot):", "Adds labeled German review examples inside the prompt."),
            ("P3 (Dev-Optimized 10-Shot):", "Selects the 10 most informative examples covering rare categories (~1,760 tokens/doc)."),
            ("High Inference Cost:", "2.7x more input tokens on every test document."),
        ],
        bg_color=BG_BLUE,
        accent_color=BLUE,
    )

    add_card(
        s4,
        left=8.8,
        top=1.6,
        width=3.8,
        height=5.2,
        title="3. Fine-Tuning (LoRA/QLoRA)",
        bullets=[
            ("Trains on 19,432 Docs:", "Adapts 0.79% of weights (20.8M params) for 3 epochs."),
            ("L1 (16-bit LoRA):", "Full 16-bit base weights (7.2 GB VRAM)."),
            ("Q1 (4-bit QLoRA):", "Compressed 4-bit base weights (2.9 GB VRAM — runs on any Colab GPU)."),
        ],
        bg_color=BG_TEAL,
        accent_color=TEAL,
    )

    # =========================================================================
    # SLIDE 5: MAIN RESULTS — TASK C BENCHMARK
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(
        s5,
        "4. Main Results: Task C Aspect–Polarity Classification",
        "Fine-tuning outperforms the best 10-shot prompt by +14.5 Micro-F1 (+19.5 on Official Java-Compat F1)",
        5,
        total_slides,
    )

    rows5, cols5 = 8, 6
    t5_shape = s5.shapes.add_table(rows5, cols5, Inches(0.7), Inches(1.55), Inches(11.933), Inches(4.1))
    t5 = t5_shape.table
    t5.columns[0].width = Inches(3.233)
    t5.columns[1].width = Inches(1.8)
    t5.columns[2].width = Inches(1.7)
    t5.columns[3].width = Inches(1.8)
    t5.columns[4].width = Inches(1.7)
    t5.columns[5].width = Inches(1.7)

    h5 = [
        "Approach (test_syn = 2,566 docs)",
        "C1 Strict F1 (Category)",
        "C2 Strict F1 (Cat+Pol)",
        "C2 Official Java F1",
        "Exact Document Match",
        "Diachronic Test C2 F1",
    ]
    d5 = [
        ["P0 Minimal Zero-Shot", "0.555", "0.464", "0.302", "36.8%", "0.435"],
        ["P1 Guideline Zero-Shot", "0.668", "0.587", "0.416", "47.4%", "0.555"],
        ["P2 Few-Shot (2 Examples)", "0.706", "0.639", "0.469", "53.4%", "0.608"],
        ["P2 Few-Shot (5 Examples)", "0.741", "0.667", "0.501", "57.3%", "0.638"],
        ["P3 Optimized 10-Shot", "0.767", "0.703", "0.542", "60.0%", "0.711"],
        ["Q1 4-bit QLoRA Fine-Tuned", "0.873", "0.836", "0.718", "75.3%", "0.823"],
        ["L1 16-bit LoRA Fine-Tuned", "0.882", "0.849", "0.737", "76.8%", "0.818"],
    ]

    for c_i, text in enumerate(h5):
        c = t5.cell(0, c_i)
        c.fill.solid()
        c.fill.fore_color.rgb = NAVY
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = c.text_frame.paragraphs[0]
        p.text = text
        p.font.bold = True
        p.font.size = Pt(12.5)
        p.font.color.rgb = WHITE
        p.font.name = "Calibri"

    for r_i, r_vals in enumerate(d5, start=1):
        is_ft = r_i >= 6
        for c_i, val in enumerate(r_vals):
            c = t5.cell(r_i, c_i)
            c.fill.solid()
            c.fill.fore_color.rgb = BG_TEAL if is_ft else (WHITE if r_i % 2 == 1 else BG_LIGHT)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = c.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(12.5)
            p.font.bold = is_ft or c_i == 0
            p.font.color.rgb = TEAL if (is_ft and c_i >= 1) else NAVY
            p.font.name = "Calibri"

    add_card(
        s5,
        left=0.7,
        top=5.85,
        width=11.933,
        height=1.1,
        title="Three Key Findings from the Benchmark Table",
        bullets=[
            ("1. Guidelines help Zero-Shot (+12.3 F1) and 10-Shot helps further (+11.6 F1), but Prompting plateaus at 0.703.", ""),
            ("2. Fine-Tuning jumps to 0.849 F1 (predicting the exact full set of opinions on 76.8% of documents vs. 60.0% for 10-shot).", ""),
        ],
        bg_color=BG_AMBER,
        accent_color=AMBER,
        title_size=14,
        body_size=12.5,
    )

    # =========================================================================
    # SLIDE 6: HOW MUCH DATA DO WE NEED? & WHERE DO GAINS COME FROM?
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(
        s6,
        "5. Sample Efficiency & Per-Category Improvements",
        "How many labeled reviews are needed to beat prompting, and which categories improve most?",
        6,
        total_slides,
    )

    lc_img = Path("results/figures/learning_curve_task_c2.png")
    err_img = Path("results/figures/error_analysis_delta_f1.png")
    if lc_img.exists():
        s6.shapes.add_picture(str(lc_img), Inches(0.7), Inches(1.55), width=Inches(5.8))
    if err_img.exists():
        s6.shapes.add_picture(str(err_img), Inches(6.8), Inches(1.55), width=Inches(5.8))

    add_card(
        s6,
        left=0.7,
        top=5.35,
        width=5.8,
        height=1.6,
        title="Sample Efficiency (Left Chart)",
        bullets=[
            ("1,000 docs (5% of train):", "Reaches 0.701 F1 in 1.6 GPU minutes — matching 10-shot prompting."),
            ("2,000+ docs:", "Clearly surpasses all prompting baselines."),
        ],
        bg_color=BG_BLUE,
        accent_color=BLUE,
        title_size=15,
        body_size=13,
    )

    add_card(
        s6,
        left=6.8,
        top=5.35,
        width=5.8,
        height=1.6,
        title="Category Gains & Span Extraction (Right Chart)",
        bullets=[
            ("All 19 Categories Improve:", "Largest gains on subtle/implicit categories (Allgemein, Zugfahrt, Sicherheit)."),
            ("Task D Target Extraction:", "Exact span F1 jumps from 0.566 (5-shot) to 0.762 (QLoRA, +19.6 F1)."),
        ],
        bg_color=BG_TEAL,
        accent_color=TEAL,
        title_size=15,
        body_size=13,
    )

    # =========================================================================
    # SLIDE 7: EXECUTIVE SUMMARY & PRACTICAL RECOMMENDATIONS
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(
        s7,
        "6. Main Takeaways of the Experiment",
        "When to use Prompting vs. Fine-Tuning for domain-specific classification & extraction",
        7,
        total_slides,
    )

    add_card(
        s7,
        left=0.7,
        top=1.6,
        width=3.8,
        height=5.2,
        title="1. Prompting Hits a Ceiling",
        bullets=[
            ("Good for Prototyping:", "Adding official guidelines (P1) and 10 examples (P3) lifts F1 from 0.464 to 0.703."),
            ("Cannot Master Boundaries:", "Even 10 examples cannot capture all domain nuances across 19 categories and noisy German posts."),
            ("Expensive at Scale:", "2.7x higher token cost on every single inference request."),
        ],
        bg_color=BG_LIGHT,
        accent_color=SLATE_DARK,
    )

    add_card(
        s7,
        left=4.75,
        top=1.6,
        width=3.8,
        height=5.2,
        title="2. Fine-Tuning Wins Big",
        bullets=[
            ("+14.5 F1 Accuracy Jump:", "Reaches 0.849 Strict Micro-F1 on Task C2 and +19.6 F1 on Task D span extraction."),
            ("High Data Efficiency:", "Just 1,000 labeled reviews match 10-shot prompting; 2,000+ beat it decisively."),
            ("Robust Over Time:", "Maintains 0.823 F1 on the diachronic test set collected months later."),
        ],
        bg_color=BG_TEAL,
        accent_color=TEAL,
    )

    add_card(
        s7,
        left=8.8,
        top=1.6,
        width=3.8,
        height=5.2,
        title="3. 4-Bit QLoRA = Best ROI",
        bullets=[
            ("98.5% of 16-Bit Accuracy:", "4-bit QLoRA achieves 0.836 F1 vs. 0.849 F1 for 16-bit LoRA."),
            ("Runs on Any GPU (2.9 GB):", "Fits easily on a free Google Colab T4 or L4 GPU."),
            ("Fast & CompactInference:", "Uses compact prompts at test time with 100.0% valid JSON output."),
        ],
        bg_color=BG_AMBER,
        accent_color=AMBER,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))
    return out


if __name__ == "__main__":
    p1 = build_core_concepts_presentation()
    p2 = build_task_and_results_presentation()
    print(f"Saved Deck 1: {p1} ({p1.stat().st_size / 1024:.1f} KB)")
    print(f"Saved Deck 2: {p2} ({p2.stat().st_size / 1024:.1f} KB)")
