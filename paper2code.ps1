# paper2code.ps1 — fully automatic end-to-end pipeline
#
# Usage (called by paper2code.bat):
#   paper2code.bat https://arxiv.org/pdf/2510.01193
#   paper2code.bat https://arxiv.org/abs/2510.01193
#   paper2code.bat C:\papers\mypaper.pdf
#   paper2code.bat https://arxiv.org/pdf/2510.01193 --provider groq --model llama-3.3-70b-versatile
#   paper2code.bat https://arxiv.org/pdf/2510.01193 --api_key sk-... --output C:\myoutputs
#
# Options:
#   --provider   openai|groq|cerebras|openrouter|mistral|github|sambanova|gemini|cohere|cloudflare
#   --model      model name for the chosen provider
#   --api_key    explicit API key (overrides env var)
#   --output     root output directory (default: .\outputs)
#   --latex      path to a .tex file to use instead of PDF (skips PDF download)
#   --debug      run the debugging agent if reproduce.ps1 exits non-zero
#   --eval       run reference-free evaluation after coding

param(
    [Parameter(Position=0)]
    [string]$Source = "",

    [string]$Provider  = "openai",
    [string]$Model     = "",
    [string]$ApiKey    = "",
    [string]$Output    = "",
    [string]$Latex     = "",
    [switch]$RunDebug,
    [switch]$Eval
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function Info  ($msg) { Write-Host "  $msg" -ForegroundColor Cyan }
function Good  ($msg) { Write-Host "  [OK] $msg" -ForegroundColor Green }
function Warn  ($msg) { Write-Host "  [WARN] $msg" -ForegroundColor Yellow }
function Fail  ($msg) { Write-Host "  [ERROR] $msg" -ForegroundColor Red; exit 1 }
function Banner($msg) { Write-Host "`n========================================" -ForegroundColor Magenta
                        Write-Host "  $msg" -ForegroundColor Magenta
                        Write-Host "========================================" -ForegroundColor Magenta }

function Require-Python {
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        Fail "Python not found. Activate your conda env first:  conda activate papertocode"
    }
}

function Check-Module ($module) {
    $r = python -c "import $module; print('ok')" 2>&1
    return ("$r" -match "ok")
}

# ---------------------------------------------------------------------------
# Resolve root directory (where this script lives)
# ---------------------------------------------------------------------------
$Root = $PSScriptRoot
if (-not $Root) { $Root = (Get-Location).Path }

$CodesDir = Join-Path $Root "codes"

# ---------------------------------------------------------------------------
# Parse --key value pairs that bat passes as positional args
# (PowerShell param() already handled named args; these catch the rest)
# ---------------------------------------------------------------------------
# (all handled by param block above)

# ---------------------------------------------------------------------------
# Validate input
# ---------------------------------------------------------------------------
if (-not $Source) {
    Write-Host @"

Usage:
  paper2code.bat <PDF_URL_or_PATH> [options]

Examples:
  paper2code.bat https://arxiv.org/pdf/2510.01193
  paper2code.bat https://arxiv.org/abs/2510.01193
  paper2code.bat C:\papers\mypaper.pdf --provider groq --model llama-3.3-70b-versatile
  paper2code.bat https://arxiv.org/pdf/2510.01193 --provider openai --model o3-mini --output C:\myoutputs

Options:
  --provider   LLM provider  (default: openai)
  --model      Model name    (default: o3-mini for openai)
  --api_key    Explicit API key
  --output     Output root   (default: .\outputs)
  --latex      Path to .tex file to use instead of PDF
  --rundebug   Run debugging agent if generated code fails
  --eval       Run reference-free evaluation after coding

"@
    exit 0
}

Require-Python

# Default model per provider
if (-not $Model) {
    $defaults = @{
        openai      = "o3-mini"
        groq        = "llama-3.3-70b-versatile"
        cerebras    = "llama-3.3-70b"
        openrouter  = "deepseek/deepseek-r1:free"
        mistral     = "mistral-large-latest"
        github      = "gpt-4.1"
        sambanova   = "Meta-Llama-3.3-70B-Instruct"
        gemini      = "gemini-2.5-flash"
        cohere      = "command-r-plus"
        cloudflare  = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
    }
    $Model = $defaults[$Provider.ToLower()]
    if (-not $Model) { $Model = "o3-mini" }
}

# Output root
if (-not $Output) { $Output = Join-Path $Root "outputs" }

# API key flag
$ApiKeyFlag = @()
if ($ApiKey) { $ApiKeyFlag = @("--api_key", $ApiKey) }

# ---------------------------------------------------------------------------
# Step 1 — Resolve PDF
# ---------------------------------------------------------------------------
Banner "Step 1 — Resolving input"

$PdfPath   = ""
$PaperName = ""
$UseLatex  = $false

if ($Latex) {
    # User supplied a .tex file directly — skip PDF download
    if (-not (Test-Path $Latex)) { Fail "LaTeX file not found: $Latex" }
    $UseLatex  = $true
    $PaperName = [System.IO.Path]::GetFileNameWithoutExtension($Latex)
    Info "Using LaTeX source: $Latex"
    Info "Paper name: $PaperName"
}
elseif ($Source -match "^https?://") {
    # --- URL input ---

    # Normalise arxiv abstract URL -> direct PDF URL
    # e.g. https://arxiv.org/abs/2510.01193 -> https://arxiv.org/pdf/2510.01193
    $Url = $Source -replace "arxiv\.org/abs/", "arxiv.org/pdf/"
    # Strip trailing version tags like v1, v2 for cleaner naming
    $UrlStem = $Url -replace "v\d+$", ""

    # Derive a safe paper name from the URL
    $LastSegment = ($Url.TrimEnd("/").Split("/"))[-1] -replace "\?.*$", "" -replace "v\d+$", ""
    $PaperName = $LastSegment -replace "[^\w\-]", "_"
    if (-not $PaperName) { $PaperName = "paper" }

    # Download PDF
    $TmpDir  = Join-Path $env:TEMP "paper2code_$PaperName"
    New-Item -ItemType Directory -Force -Path $TmpDir | Out-Null
    $PdfPath = Join-Path $TmpDir "$PaperName.pdf"

    if (Test-Path $PdfPath) {
        Good "PDF already cached: $PdfPath"
    } else {
        Info "Downloading PDF from: $Url"
        try {
            Invoke-WebRequest -Uri $Url -OutFile $PdfPath -UserAgent "Mozilla/5.0" -TimeoutSec 60
            Good "Downloaded -> $PdfPath"
        } catch {
            Fail "Failed to download PDF: $_"
        }
    }

    # Verify it's actually a PDF
    $Header = [System.IO.File]::ReadAllBytes($PdfPath) | Select-Object -First 4
    $Magic  = [System.Text.Encoding]::ASCII.GetString($Header)
    if ($Magic -ne "%PDF") {
        # arxiv sometimes redirects to an HTML page — try appending .pdf
        if ($Url -notmatch "\.pdf$") {
            $Url2 = $Url + ".pdf"
            Info "Retrying with URL: $Url2"
            try {
                Invoke-WebRequest -Uri $Url2 -OutFile $PdfPath -UserAgent "Mozilla/5.0" -TimeoutSec 60
                $Header = [System.IO.File]::ReadAllBytes($PdfPath) | Select-Object -First 4
                $Magic  = [System.Text.Encoding]::ASCII.GetString($Header)
            } catch {}
        }
        if ($Magic -ne "%PDF") {
            Fail "Downloaded file is not a valid PDF (header: '$Magic'). Check the URL."
        }
    }

    Info "Paper name: $PaperName"
}
else {
    # --- Local file input ---
    if (-not (Test-Path $Source)) { Fail "File not found: $Source" }
    $PdfPath   = (Resolve-Path $Source).Path
    $PaperName = [System.IO.Path]::GetFileNameWithoutExtension($PdfPath) -replace "[^\w\-]", "_"
    Info "Using local PDF: $PdfPath"
    Info "Paper name: $PaperName"
}

# ---------------------------------------------------------------------------
# Step 2 — Convert PDF -> JSON (skip if using LaTeX)
# ---------------------------------------------------------------------------
$WorkDir    = Join-Path $Output $PaperName
$RawJson    = Join-Path $WorkDir "${PaperName}_raw.json"
$CleanJson  = Join-Path $WorkDir "${PaperName}_cleaned.json"
$OutputRepo = Join-Path $Output  "${PaperName}_repo"

New-Item -ItemType Directory -Force -Path $WorkDir    | Out-Null
New-Item -ItemType Directory -Force -Path $OutputRepo | Out-Null

if (-not $UseLatex) {
    Banner "Step 2 — PDF -> JSON"

    if (-not (Check-Module "fitz")) {
        Warn "PyMuPDF not found. Installing..."
        python -m pip install pymupdf --quiet
    }

    Info "Converting PDF to JSON..."
    python (Join-Path $CodesDir "pdf_to_json.py") `
        --pdf_path         $PdfPath `
        --output_json_path $RawJson
    if ($LASTEXITCODE -ne 0) { Fail "pdf_to_json.py failed" }

    Banner "Step 3 — Cleaning JSON"
    python (Join-Path $CodesDir "0_pdf_process.py") `
        --input_json_path  $RawJson `
        --output_json_path $CleanJson
    if ($LASTEXITCODE -ne 0) { Fail "0_pdf_process.py failed" }

    Good "Cleaned JSON -> $CleanJson"
} else {
    Banner "Step 2+3 — LaTeX source (no PDF conversion needed)"
    $CleanJson = ""   # not used in latex mode
    Good "Will use LaTeX file: $Latex"
}

# ---------------------------------------------------------------------------
# Step 4 — Planning
# ---------------------------------------------------------------------------
Banner "Step 4 — Planning"

if ($UseLatex) {
    python (Join-Path $CodesDir "1_planning.py") `
        --paper_name     $PaperName `
        --gpt_version    $Model `
        --paper_format   LaTeX `
        --pdf_latex_path $Latex `
        --output_dir     $WorkDir `
        --provider       $Provider `
        @ApiKeyFlag
} else {
    python (Join-Path $CodesDir "1_planning.py") `
        --paper_name    $PaperName `
        --gpt_version   $Model `
        --pdf_json_path $CleanJson `
        --output_dir    $WorkDir `
        --provider      $Provider `
        @ApiKeyFlag
}
if ($LASTEXITCODE -ne 0) { Fail "1_planning.py failed" }
Good "Planning complete"

# ---------------------------------------------------------------------------
# Step 5 — Extract config
# ---------------------------------------------------------------------------
Banner "Step 5 — Extracting config"

python (Join-Path $CodesDir "1.1_extract_config.py") `
    --paper_name $PaperName `
    --output_dir $WorkDir
if ($LASTEXITCODE -ne 0) { Warn "1.1_extract_config.py failed (non-fatal)" }

$ConfigSrc = Join-Path $WorkDir "planning_config.yaml"
$ConfigDst = Join-Path $OutputRepo "config.yaml"
if (Test-Path $ConfigSrc) {
    Copy-Item -Force $ConfigSrc $ConfigDst
    Good "config.yaml -> $ConfigDst"
}

# ---------------------------------------------------------------------------
# Step 6 — Analyzing
# ---------------------------------------------------------------------------
Banner "Step 6 — Analyzing"

if ($UseLatex) {
    python (Join-Path $CodesDir "2_analyzing.py") `
        --paper_name     $PaperName `
        --gpt_version    $Model `
        --paper_format   LaTeX `
        --pdf_latex_path $Latex `
        --output_dir     $WorkDir `
        --provider       $Provider `
        @ApiKeyFlag
} else {
    python (Join-Path $CodesDir "2_analyzing.py") `
        --paper_name    $PaperName `
        --gpt_version   $Model `
        --pdf_json_path $CleanJson `
        --output_dir    $WorkDir `
        --provider      $Provider `
        @ApiKeyFlag
}
if ($LASTEXITCODE -ne 0) { Fail "2_analyzing.py failed" }
Good "Analysis complete"

# ---------------------------------------------------------------------------
# Step 7 — Coding
# ---------------------------------------------------------------------------
Banner "Step 7 — Coding"

if ($UseLatex) {
    python (Join-Path $CodesDir "3_coding.py") `
        --paper_name     $PaperName `
        --gpt_version    $Model `
        --paper_format   LaTeX `
        --pdf_latex_path $Latex `
        --output_dir     $WorkDir `
        --output_repo_dir $OutputRepo `
        --provider       $Provider `
        @ApiKeyFlag
} else {
    python (Join-Path $CodesDir "3_coding.py") `
        --paper_name     $PaperName `
        --gpt_version    $Model `
        --pdf_json_path  $CleanJson `
        --output_dir     $WorkDir `
        --output_repo_dir $OutputRepo `
        --provider       $Provider `
        @ApiKeyFlag
}
if ($LASTEXITCODE -ne 0) { Fail "3_coding.py failed" }
Good "Coding complete"

# ---------------------------------------------------------------------------
# Step 8 — (Optional) Run reproduce.ps1
# ---------------------------------------------------------------------------
$ReproScript = Join-Path $OutputRepo "reproduce.ps1"
$ReproFailed = $false

if (Test-Path $ReproScript) {
    Banner "Step 8 — Running reproduce.ps1"
    Push-Location $OutputRepo
    try {
        powershell.exe -NoProfile -ExecutionPolicy Bypass -File "reproduce.ps1"
        if ($LASTEXITCODE -ne 0) {
            Warn "reproduce.ps1 exited with code $LASTEXITCODE"
            $ReproFailed = $true
        } else {
            Good "reproduce.ps1 succeeded"
        }
    } catch {
        Warn "reproduce.ps1 threw an exception: $_"
        $ReproFailed = $true
    } finally {
        Pop-Location
    }
} else {
    Info "No reproduce.ps1 found — skipping auto-run"
}

# ---------------------------------------------------------------------------
# Step 9 — (Optional) Debugging agent
# ---------------------------------------------------------------------------
if ($RunDebug -and $ReproFailed) {
    Banner "Step 9 — Debugging agent"

    $ErrorFile = Join-Path $OutputRepo "error.txt"
    if (-not (Test-Path $ErrorFile)) {
        # Capture stderr from a fresh reproduce attempt
        $ErrorLog = powershell.exe -NoProfile -ExecutionPolicy Bypass `
            -File (Join-Path $OutputRepo "reproduce.ps1") 2>&1
        $ErrorLog | Out-File $ErrorFile -Encoding utf8
    }

    python (Join-Path $CodesDir "4_debugging.py") `
        --paper_name      $PaperName `
        --model           $Model `
        --provider        $Provider `
        --error_file_name $ErrorFile `
        --output_dir      $WorkDir `
        --output_repo_dir $OutputRepo `
        --save_num        1 `
        @ApiKeyFlag

    if ($LASTEXITCODE -eq 0) { Good "Debugging complete" } else { Warn "Debugging agent returned errors" }
}

# ---------------------------------------------------------------------------
# Step 10 — (Optional) Evaluation
# ---------------------------------------------------------------------------
if ($Eval) {
    Banner "Step 10 — Reference-free evaluation"

    $EvalArgs = @(
        "--paper_name",      $PaperName,
        "--pdf_json_path",   $CleanJson,
        "--data_dir",        (Join-Path $Root "data"),
        "--output_dir",      $WorkDir,
        "--target_repo_dir", $OutputRepo,
        "--eval_result_dir", (Join-Path $Root "results"),
        "--eval_type",       "ref_free",
        "--generated_n",     "8",
        "--gpt_version",     $Model,
        "--provider",        $Provider,
        "--papercoder"
    )
    python (Join-Path $CodesDir "eval.py") @EvalArgs @ApiKeyFlag
}

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
Banner "Done"
Good "Paper name : $PaperName"
Good "Output dir : $WorkDir"
Good "Repo dir   : $OutputRepo"
if (Test-Path $CleanJson) { Good "Cleaned JSON: $CleanJson" }
Write-Host ""
