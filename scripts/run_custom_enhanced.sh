#!/bin/bash

# API key is already in environment

GPT_VERSION="o3-2025-04-16"
IMAGE_GPT_VERSION="o4-mini-2025-04-16"

PAPER_NAME="Segar"
PDF_PATH="/Users/Lordof44/Projects/segar-et-al-development-and-validation-of-machine-learning-based-race-specific-models-to-predict-10-year-risk-of-heart.pdf"
CUSTOM_DIR="/Users/Lordof44/Documents/GitHub/Paper2Code/custom_paper"
PDF_JSON_PATH="${CUSTOM_DIR}/paper.json"
PDF_JSON_CLEANED_PATH="${CUSTOM_DIR}/paper_cleaned.json"
ENHANCED_JSON_PATH="${CUSTOM_DIR}/enhanced_paper.json"
OUTPUT_DIR="/Users/Lordof44/Documents/GitHub/Paper2Code/outputs/Segar_enhanced"
OUTPUT_REPO_DIR="/Users/Lordof44/Documents/GitHub/Paper2Code/outputs/Segar_repo_enhanced"

mkdir -p $CUSTOM_DIR
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_REPO_DIR

echo $PAPER_NAME

# Copy the paper to the custom directory
echo "------- Copying Paper -------"
cp "$PDF_PATH" "${CUSTOM_DIR}/paper.pdf"
PDF_PATH="${CUSTOM_DIR}/paper.pdf"

# First, we need to run the GROBID service to process PDF
echo "------- Starting GROBID -------"
echo "IMPORTANT: Make sure GROBID is running in another terminal with the command:"
echo "cd \$HOME/grobid-0.7.3 && ./gradlew run"
echo "Press Enter when GROBID is running..."
read -p ""

echo "------- Processing PDF -------"
cd /Users/Lordof44/Documents/GitHub/Paper2Code
source paper2code_env/bin/activate
python s2orc-doc2json/doc2json/grobid2json/process_pdf.py -i "$PDF_PATH" -t "${CUSTOM_DIR}/temp_dir/" -o "${CUSTOM_DIR}/"

# Check if PDF processing was successful
if [ ! -f "${PDF_JSON_PATH}" ]; then
    echo "ERROR: PDF processing failed. Make sure GROBID is running properly."
    exit 1
fi

echo "------- Preprocess -------"
python codes/0_pdf_process.py \
    --input_json_path ${PDF_JSON_PATH} \
    --output_json_path ${PDF_JSON_CLEANED_PATH}

echo "------- Extracting Figures and Getting LLM Descriptions -------"
# Install PyMuPDF if not already installed
pip install PyMuPDF

# Run the figure extraction script
python codes/extract_figures.py \
    --pdf_path "$PDF_PATH" \
    --json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${CUSTOM_DIR} \
    --gpt_version ${IMAGE_GPT_VERSION}

# Use the enhanced JSON for the rest of the pipeline
if [ -f "$ENHANCED_JSON_PATH" ]; then
    echo "Using enhanced JSON with figure descriptions"
    PDF_JSON_CLEANED_PATH=${ENHANCED_JSON_PATH}
else
    echo "WARNING: Enhanced JSON not found, using regular cleaned JSON"
fi

echo "------- PaperCoder -------"
python codes/1_planning.py \
    --paper_name $PAPER_NAME \
    --gpt_version ${GPT_VERSION} \
    --pdf_json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${OUTPUT_DIR}

python codes/1.1_extract_config.py \
    --paper_name $PAPER_NAME \
    --output_dir ${OUTPUT_DIR}

cp -rp ${OUTPUT_DIR}/planning_config.yaml ${OUTPUT_REPO_DIR}/config.yaml

python codes/2_analyzing.py \
    --paper_name $PAPER_NAME \
    --gpt_version ${GPT_VERSION} \
    --pdf_json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${OUTPUT_DIR}

python codes/3_coding.py  \
    --paper_name $PAPER_NAME \
    --gpt_version ${GPT_VERSION} \
    --pdf_json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --output_repo_dir ${OUTPUT_REPO_DIR}