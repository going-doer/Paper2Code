# Paper2Code - Windows Edition

> A Windows-native application of [Paper2Code](https://github.com/going-doer/Paper2Code) with one-command automation and support for 10 LLM providers.
>
> Original research by **Minju Seo, Jinheon Baek, Seongyun Lee, and Sung Ju Hwang** — ICLR 2026.
> [Read the paper](https://arxiv.org/abs/2504.17192) · [Dataset](https://huggingface.co/datasets/iaminju/paper2code) · [Original repo](https://github.com/going-doer/Paper2Code)

![PaperCoder Overview](./assets/papercoder_overview.png)

**PaperCoder** is a multi-agent LLM system that transforms a machine-learning research paper into a working code repository through three specialised stages: **planning -> analysis -> code generation**.

This fork makes the full pipeline run natively on **Windows** with a single command — no WSL, no GROBID, no manual steps.

---

## Table of Contents

- [What's New in This Fork](#whats-new-in-this-fork)
- [Quick Start](#quick-start)
- [Supported Providers](#supported-providers)
- [Setting API Keys](#setting-api-keys)
- [All Options](#all-options)
- [Setup](#setup)
- [Running Individual Stages](#running-individual-stages)
- [Debugging a Generated Repo](#debugging-a-generated-repo)
- [Evaluation](#evaluation)
- [Benchmark Dataset](#benchmark-dataset)
- [Test Suite](#test-suite)
- [Credits](#credits)

---

## What's New in This Fork

| Area | Change |
|---|---|
| **One-command launcher** | ``paper2code.bat <URL or PDF>`` runs the entire pipeline end-to-end |
| **Automatic PDF download** | Pass an arXiv URL and the PDF is downloaded automatically |
| **No GROBID required** | PDF-to-JSON conversion uses PyMuPDF (pure Python, no Java or Docker) |
| **Windows native** | All shell scripts rewritten as PowerShell; no WSL needed |
| **10 LLM providers** | Unified ``providers.py`` abstraction — swap provider with one flag |
| **Free tier support** | Groq, Cerebras, OpenRouter, Mistral, GitHub Models, SambaNova, Gemini, Cohere, Cloudflare |
| **Path and encoding fixes** | All file I/O uses ``os.path.join`` and ``encoding='utf-8'`` |
| **Auto-debugging** | ``--rundebug`` flag runs the debugging agent if generated code fails |
| **Test suite** | ``test_suite.py`` — 41 automated tests, no API key required |

---

## Quick Start

### 1. Set up the environment

```powershell
conda create -n papertocode python=3.11
conda activate papertocode
pip install -r requirements.txt
```

### 2. Set your API key

```powershell
# OpenRouter (free models available)
$env:OPENROUTER_API_KEY = "your-key"

# Or OpenAI
$env:OPENAI_API_KEY = "your-key"

# Or any other provider — see the full list below
```

### 3. Run

```powershell
.\paper2code.bat https://arxiv.org/pdf/2510.01193 --provider openrouter --model inclusionai/ling-2.6-flash:free
```

That single command:
1. Downloads the PDF from the URL
2. Converts it to JSON (no GROBID needed)
3. Runs planning, analysis, and code generation
4. Outputs a ready-to-run repo to ``outputs/<paper_id>_repo/``

> **Note:** In PowerShell always prefix with ``.\`` — e.g. ``.\paper2code.bat``, not ``paper2code.bat``.

---

## Supported Providers

| Provider | Free tier | Env variable | Signup |
|---|---|---|---|
| ``openai`` | No | ``OPENAI_API_KEY`` | [platform.openai.com](https://platform.openai.com) |
| ``openrouter`` | Yes | ``OPENROUTER_API_KEY`` | [openrouter.ai](https://openrouter.ai) |
| ``groq`` | Yes | ``GROQ_API_KEY`` | [console.groq.com](https://console.groq.com) |
| ``cerebras`` | Yes | ``CEREBRAS_API_KEY`` | [cloud.cerebras.ai](https://cloud.cerebras.ai) |
| ``gemini`` | Yes | ``GEMINI_API_KEY`` | [aistudio.google.com](https://aistudio.google.com) |
| ``mistral`` | Yes | ``MISTRAL_API_KEY`` | [console.mistral.ai](https://console.mistral.ai) |
| ``github`` | Yes* | ``GITHUB_TOKEN`` | [github.com/marketplace/models](https://github.com/marketplace/models) |
| ``sambanova`` | Yes | ``SAMBANOVA_API_KEY`` | [cloud.sambanova.ai](https://cloud.sambanova.ai) |
| ``cohere`` | Yes | ``COHERE_API_KEY`` | [dashboard.cohere.com](https://dashboard.cohere.com) |
| ``cloudflare`` | Yes | ``CLOUDFLARE_API_KEY`` + ``CLOUDFLARE_ACCOUNT_ID`` | [dash.cloudflare.com](https://dash.cloudflare.com) |

---

## Setting API Keys

Set the environment variable for the provider you want to use before running the pipeline.

```powershell
$env:OPENAI_API_KEY          = "sk-..."
$env:OPENROUTER_API_KEY      = "sk-or-..."
$env:GROQ_API_KEY            = "gsk_..."
$env:CEREBRAS_API_KEY        = "..."
$env:GEMINI_API_KEY          = "..."
$env:MISTRAL_API_KEY         = "..."
$env:GITHUB_TOKEN            = "ghp_..."
$env:SAMBANOVA_API_KEY       = "..."
$env:COHERE_API_KEY          = "..."
$env:CLOUDFLARE_API_KEY      = "..."
$env:CLOUDFLARE_ACCOUNT_ID   = "..."   # Cloudflare requires both
```

Alternatively, pass the key directly at runtime:

```powershell
.\paper2code.bat https://arxiv.org/pdf/2510.01193 --provider openai --model o3-mini --api_key sk-...
```

---

## All Options

```
.\paper2code.bat <URL_or_PDF_path> [options]
```

| Option | Default | Description |
|---|---|---|
| ``--provider`` | ``openai`` | LLM provider to use |
| ``--model`` | per-provider default | Model name for the chosen provider |
| ``--api_key`` | from env var | Explicit API key (overrides env var) |
| ``--output`` | ``.\outputs`` | Root directory for all outputs |
| ``--latex`` | — | Path to a ``.tex`` file to use instead of PDF |
| ``--rundebug`` | off | Run the debugging agent if ``reproduce.ps1`` fails |
| ``--eval`` | off | Run reference-free evaluation after coding |

### Examples

```powershell
# arXiv URL — PDF downloaded automatically
.\paper2code.bat https://arxiv.org/pdf/2510.01193

# Abstract URL also works
.\paper2code.bat https://arxiv.org/abs/2510.01193 --provider groq --model llama-3.3-70b-versatile

# Local PDF
.\paper2code.bat C:\papers\mypaper.pdf --provider gemini --model gemini-2.5-flash

# Custom output directory
.\paper2code.bat https://arxiv.org/pdf/2510.01193 --provider openai --model o3-mini --output C:\myoutputs

# With auto-debugging and evaluation
.\paper2code.bat https://arxiv.org/pdf/2510.01193 --provider openrouter --model deepseek/deepseek-r1:free --rundebug --eval

# LaTeX source instead of PDF
.\paper2code.bat --latex examples\Transformer_cleaned.tex --provider openai --model gpt-4o
```

### Output structure

```
outputs/
+-- <paper_id>/
|   +-- planning_artifacts/
|   +-- analyzing_artifacts/
|   +-- coding_artifacts/
+-- <paper_id>_repo/          <- final generated repository
```

---

## Setup

### Requirements

- Windows 10 or 11
- [Anaconda](https://www.anaconda.com) or Python 3.10+
- PowerShell 5.1+ (built into Windows)

### Installation

```powershell
conda create -n papertocode python=3.11
conda activate papertocode
pip install -r requirements.txt
```

> ``vllm`` (local open-source models) is **Linux/CUDA only** and is excluded from Windows installs. Use any of the cloud providers above instead.

---

## Running Individual Stages

``paper2code.bat`` runs all stages automatically. To run them individually:

```powershell
# Stage 0 — Clean a raw JSON
python codes/0_pdf_process.py `
    --input_json_path  examples/Transformer.json `
    --output_json_path examples/Transformer_cleaned.json

# Stage 1 — Planning
python codes/1_planning.py `
    --paper_name    Transformer `
    --gpt_version   o3-mini `
    --provider      openai `
    --pdf_json_path examples/Transformer_cleaned.json `
    --output_dir    outputs

# Stage 2 — Analyzing
python codes/2_analyzing.py `
    --paper_name    Transformer `
    --gpt_version   o3-mini `
    --provider      openai `
    --pdf_json_path examples/Transformer_cleaned.json `
    --output_dir    outputs

# Stage 3 — Coding
python codes/3_coding.py `
    --paper_name      Transformer `
    --gpt_version     o3-mini `
    --provider        openai `
    --pdf_json_path   examples/Transformer_cleaned.json `
    --output_dir      outputs `
    --output_repo_dir outputs/Transformer_repo
```

For LaTeX input replace ``--pdf_json_path`` with ``--paper_format LaTeX --pdf_latex_path <path_to_tex>``.

---

## Debugging a Generated Repo

If the generated code fails to run, use the debugging agent:

```powershell
.\scripts\run_debug.ps1 -Provider openai
```

Or directly:

```powershell
python codes/4_debugging.py `
    --paper_name      Transformer `
    --model           o3-mini `
    --provider        openai `
    --error_file_name outputs/Transformer_repo/error.txt `
    --output_dir      outputs/Transformer `
    --output_repo_dir outputs/Transformer_repo `
    --save_num        1
```

The ``--rundebug`` flag in ``paper2code.bat`` does this automatically when ``reproduce.ps1`` exits with an error.

---

## Evaluation

Evaluate a generated repository with a 1-5 correctness score averaged over ``n`` judge-model samples.

```powershell
# Reference-free (no gold repo needed)
python codes/eval.py `
    --paper_name      Transformer `
    --pdf_json_path   examples/Transformer_cleaned.json `
    --data_dir        data `
    --output_dir      outputs/Transformer `
    --target_repo_dir outputs/Transformer_repo `
    --eval_result_dir results `
    --eval_type       ref_free `
    --generated_n     8 `
    --provider        openai `
    --gpt_version     o3-mini `
    --papercoder

# Reference-based (requires a gold repo)
python codes/eval.py `
    --paper_name      Transformer `
    --pdf_json_path   examples/Transformer_cleaned.json `
    --data_dir        data `
    --output_dir      outputs/Transformer `
    --target_repo_dir outputs/Transformer_repo `
    --gold_repo_dir   examples/Transformer_gold_repo `
    --eval_result_dir results `
    --eval_type       ref_based `
    --generated_n     8 `
    --provider        openai `
    --gpt_version     o3-mini `
    --papercoder
```

``--provider`` and ``--gpt_version`` control which model acts as judge — any provider from the table above works.

**Example output:**

```
========================================
  Evaluation Summary
  Paper: Transformer
  Type:  ref_based
  Score: 4.5000  (valid: 8/8)
  Cost:  $0.1645
========================================
```

---

## Benchmark Dataset

- HuggingFace: [iaminju/paper2code](https://huggingface.co/datasets/iaminju/paper2code)
- Dataset description: [data/paper2code](https://github.com/going-doer/Paper2Code/tree/main/data/paper2code)
- Details in Section 4.1 of the [paper](https://arxiv.org/abs/2504.17192)

---

## Test Suite

Verify your setup without needing an API key:

```powershell
conda activate papertocode
python test_suite.py
```

Expected output: **41/41 tests passed**.

---

## Credits

This repository is a Windows and multi-provider implementation of the original **Paper2Code** project.

**Original paper:**
> Minju Seo, Jinheon Baek, Seongyun Lee, Sung Ju Hwang.
> *"Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning"*
> International Conference on Learning Representations (ICLR), 2026.
> [arXiv:2504.17192](https://arxiv.org/abs/2504.17192) · [github.com/going-doer/Paper2Code](https://github.com/going-doer/Paper2Code)

All credit for the PaperCoder architecture, benchmark, and evaluation methodology belongs to the original authors. This fork adds Windows compatibility, one-command automation, and multi-provider support. The science is entirely theirs.
