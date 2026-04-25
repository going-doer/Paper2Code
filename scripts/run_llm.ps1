# NOTE: vllm (used by the _llm scripts) requires Linux + CUDA and does NOT support Windows natively.
# To use these scripts on Windows, run them inside WSL2 (Windows Subsystem for Linux) with a CUDA-capable GPU.
# The OpenAI-based scripts (run.ps1 / run_latex.ps1) work natively on Windows without this limitation.

$MODEL_NAME = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
$TP_SIZE = 2

$PAPER_NAME = "Transformer"
$PDF_JSON_PATH = "..\examples\Transformer.json"          # .json
$PDF_JSON_CLEANED_PATH = "..\examples\Transformer_cleaned.json"  # _cleaned.json
$OUTPUT_DIR = "..\outputs\Transformer_dscoder"
$OUTPUT_REPO_DIR = "..\outputs\Transformer_dscoder_repo"

New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $OUTPUT_REPO_DIR | Out-Null

Write-Host $PAPER_NAME

Write-Host "------- Preprocess -------"

python ..\codes\0_pdf_process.py `
    --input_json_path $PDF_JSON_PATH `
    --output_json_path $PDF_JSON_CLEANED_PATH

Write-Host "------- PaperCoder -------"

python ..\codes\1_planning_llm.py `
    --paper_name $PAPER_NAME `
    --model_name $MODEL_NAME `
    --tp_size $TP_SIZE `
    --pdf_json_path $PDF_JSON_CLEANED_PATH `
    --output_dir $OUTPUT_DIR

python ..\codes\1.1_extract_config.py `
    --paper_name $PAPER_NAME `
    --output_dir $OUTPUT_DIR

Copy-Item -Force "$OUTPUT_DIR\planning_config.yaml" "$OUTPUT_REPO_DIR\config.yaml"

python ..\codes\2_analyzing_llm.py `
    --paper_name $PAPER_NAME `
    --model_name $MODEL_NAME `
    --tp_size $TP_SIZE `
    --pdf_json_path $PDF_JSON_CLEANED_PATH `
    --output_dir $OUTPUT_DIR

python ..\codes\3_coding_llm.py `
    --paper_name $PAPER_NAME `
    --model_name $MODEL_NAME `
    --tp_size $TP_SIZE `
    --pdf_json_path $PDF_JSON_CLEANED_PATH `
    --output_dir $OUTPUT_DIR `
    --output_repo_dir $OUTPUT_REPO_DIR
