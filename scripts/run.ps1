# ---------------------------------------------------------------------------
# Provider & API key configuration
# ---------------------------------------------------------------------------
# Set PROVIDER to one of:
#   openai | groq | cerebras | openrouter | mistral | github | sambanova
#   gemini | cohere | cloudflare
#
# Free model suggestions per provider:
#   openai      : o3-mini, gpt-4.1, gpt-4o
#   groq        : llama-3.3-70b-versatile, meta-llama/llama-4-scout-17b-16e-instruct,
#                 moonshotai/kimi-k2-instruct, qwen/qwen3-32b
#   cerebras    : llama-3.3-70b, qwen3-32b, qwen3-235b, gpt-oss-120b
#   openrouter  : deepseek/deepseek-r1:free, meta-llama/llama-4-scout:free,
#                 qwen/qwen3-235b-a22b:free
#   mistral     : mistral-large-latest, mistral-small-latest, ministral-8b-latest
#   github      : gpt-4o, gpt-4.1, o3, deepseek-r1, grok-3-mini
#   sambanova   : Meta-Llama-3.3-70B-Instruct, Qwen2.5-72B-Instruct
#   gemini      : gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite-preview-06-17
#   cohere      : command-r-plus, command-a-03-2025, aya-expanse-32b
#   cloudflare  : @cf/meta/llama-3.3-70b-instruct-fp8-fast, @cf/qwen/qwq-32b
#
# The API key is read from the environment variable matching your provider:
#   OPENAI_API_KEY | GROQ_API_KEY | CEREBRAS_API_KEY | OPENROUTER_API_KEY
#   MISTRAL_API_KEY | GITHUB_TOKEN | SAMBANOVA_API_KEY | GEMINI_API_KEY
#   COHERE_API_KEY | CLOUDFLARE_API_KEY  (+ CLOUDFLARE_ACCOUNT_ID)
#
# Or pass it explicitly via --api_key below.
# ---------------------------------------------------------------------------

$PROVIDER    = "openai"
$GPT_VERSION = "o3-mini"
# $API_KEY   = ""   # uncomment to pass key explicitly instead of env var

$PAPER_NAME = "Transformer"
$PDF_JSON_PATH = "..\examples\Transformer.json"          # .json
$PDF_JSON_CLEANED_PATH = "..\examples\Transformer_cleaned.json"  # _cleaned.json
$OUTPUT_DIR = "..\outputs\Transformer"
$OUTPUT_REPO_DIR = "..\outputs\Transformer_repo"

New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $OUTPUT_REPO_DIR | Out-Null

Write-Host $PAPER_NAME

Write-Host "------- Preprocess -------"

python ..\codes\0_pdf_process.py `
    --input_json_path $PDF_JSON_PATH `
    --output_json_path $PDF_JSON_CLEANED_PATH

Write-Host "------- PaperCoder -------"

python ..\codes\1_planning.py `
    --paper_name $PAPER_NAME `
    --gpt_version $GPT_VERSION `
    --pdf_json_path $PDF_JSON_CLEANED_PATH `
    --output_dir $OUTPUT_DIR `
    --provider $PROVIDER

python ..\codes\1.1_extract_config.py `
    --paper_name $PAPER_NAME `
    --output_dir $OUTPUT_DIR

Copy-Item -Force "$OUTPUT_DIR\planning_config.yaml" "$OUTPUT_REPO_DIR\config.yaml"

python ..\codes\2_analyzing.py `
    --paper_name $PAPER_NAME `
    --gpt_version $GPT_VERSION `
    --pdf_json_path $PDF_JSON_CLEANED_PATH `
    --output_dir $OUTPUT_DIR `
    --provider $PROVIDER

python ..\codes\3_coding.py `
    --paper_name $PAPER_NAME `
    --gpt_version $GPT_VERSION `
    --pdf_json_path $PDF_JSON_CLEANED_PATH `
    --output_dir $OUTPUT_DIR `
    --output_repo_dir $OUTPUT_REPO_DIR `
    --provider $PROVIDER
