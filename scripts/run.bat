@echo off
SETLOCAL

REM Set variables
SET "GPT_VERSION=o3-mini"
SET "PAPER_NAME=Transformer"
SET "PDF_PATH=..\examples\Transformer.pdf"
SET "PDF_JSON_PATH=..\examples\Transformer.json"
SET "PDF_JSON_CLEANED_PATH=..\examples\Transformer_cleaned.json"
SET "OUTPUT_DIR=..\outputs\Transformer"
SET "OUTPUT_REPO_DIR=..\outputs\Transformer_repo"

REM Create output directories
echo Creating output directories...
mkdir "%OUTPUT_DIR%" >nul 2>&1
mkdir "%OUTPUT_REPO_DIR%" >nul 2>&1

echo.
echo Processing paper: %PAPER_NAME%
echo.

echo ------- Preprocess -------
echo Running 0_pdf_process.py...
python ..\codes\0_pdf_process.py ^
    --input_json_path "%PDF_JSON_PATH%" ^
    --output_json_path "%PDF_JSON_CLEANED_PATH%"
IF %ERRORLEVEL% NEQ 0 (
    echo Preprocessing failed. Please check for errors.
    EXIT /B %ERRORLEVEL%
)

echo.
echo ------- PaperCoder -------
echo Running 1_planning.py...
python ..\codes\1_planning.py ^
    --paper_name "%PAPER_NAME%" ^
    --gpt_version "%GPT_VERSION%" ^
    --pdf_json_path "%PDF_JSON_CLEANED_PATH%" ^
    --output_dir "%OUTPUT_DIR%"
IF %ERRORLEVEL% NEQ 0 (
    echo Planning stage failed. Please check for errors.
    EXIT /B %ERRORLEVEL%
)

echo.
echo Running 1.1_extract_config.py...
python ..\codes\1.1_extract_config.py ^
    --paper_name "%PAPER_NAME%" ^
    --output_dir "%OUTPUT_DIR%"
IF %ERRORLEVEL% NEQ 0 (
    echo Extracting config failed. Please check for errors.
    EXIT /B %ERRORLEVEL%
)

echo.
echo Copying planning_config.yaml...
copy "%OUTPUT_DIR%\planning_config.yaml" "%OUTPUT_REPO_DIR%\config.yaml" >nul
IF %ERRORLEVEL% NEQ 0 (
    echo Copying config file failed. Please check for errors.
    EXIT /B %ERRORLEVEL%
)

echo.
echo Running 2_analyzing.py...
python ..\codes\2_analyzing.py ^
    --paper_name "%PAPER_NAME%" ^
    --gpt_version "%GPT_VERSION%" ^
    --pdf_json_path "%PDF_JSON_CLEANED_PATH%" ^
    --output_dir "%OUTPUT_DIR%"
IF %ERRORLEVEL% NEQ 0 (
    echo Analyzing stage failed. Please check for errors.
    EXIT /B %ERRORLEVEL%
)

echo.
echo Running 3_coding.py...
python ..\codes\3_coding.py ^
    --paper_name "%PAPER_NAME%" ^
    --gpt_version "%GPT_VERSION%" ^
    --pdf_json_path "%PDF_JSON_CLEANED_PATH%" ^
    --output_dir "%OUTPUT_DIR%" ^
    --output_repo_dir "%OUTPUT_REPO_DIR%"
IF %ERRORLEVEL% NEQ 0 (
    echo Coding stage failed. Please check for errors.
    EXIT /B %ERRORLEVEL%
)

echo.
echo Script execution finished.
ENDLOCAL