# Usage:
#   .\run_debug.ps1 -ErrorFile "..\outputs\Transformer_repo\error.txt" -SaveNum 1
#
# Optional overrides:
#   -Provider      "groq"
#   -Model         "llama-3.3-70b-versatile"
#   -OutputDir     "..\outputs\Transformer"
#   -OutputRepoDir "..\outputs\Transformer_repo"
#   -PaperName     "Transformer"
#
# Provider choices: openai | groq | cerebras | openrouter | mistral |
#                   github | sambanova | gemini | cohere | cloudflare
# API key is read from the matching env var (e.g. GROQ_API_KEY for groq).

param(
    [Parameter(Mandatory=$true)]
    [string]$ErrorFile,

    [Parameter(Mandatory=$true)]
    [int]$SaveNum,

    [string]$Provider     = "openai",
    [string]$Model        = "o4-mini",
    [string]$OutputDir    = "..\outputs\Transformer",
    [string]$OutputRepoDir = "..\outputs\Transformer_repo",
    [string]$PaperName    = "Transformer"
)

python ..\codes\4_debugging.py `
    --error_file_name $ErrorFile `
    --output_dir $OutputDir `
    --output_repo_dir $OutputRepoDir `
    --paper_name $PaperName `
    --model $Model `
    --save_num $SaveNum `
    --provider $Provider
