# Figure Extraction and Analysis Module

This module enhances the Paper2Code pipeline by adding figure extraction and automated LLM-based figure analysis capabilities.

## Overview

The `extract_figures.py` script:
1. Extracts all figures from a scientific paper PDF
2. Uses o4-mini-2025-04-16 or other vision-capable models to analyze each figure
3. Enhances the paper's JSON representation with detailed figure descriptions
4. Enables downstream models (like o3-2025-04-16) to better understand visual content

## Usage

```bash
python extract_figures.py \
    --pdf_path "/path/to/paper.pdf" \
    --json_path "/path/to/paper_cleaned.json" \
    --output_dir "/path/to/output_directory" \
    --gpt_version "o4-mini-2025-04-16"
```

### Arguments:

- `--pdf_path`: Path to the PDF file containing figures
- `--json_path`: Path to the preprocessed JSON file from s2orc-doc2json
- `--output_dir`: Directory where the enhanced JSON and extracted figures will be saved
- `--gpt_version`: The OpenAI model to use for figure analysis (must support vision)

## How It Works

1. **Figure Extraction**: Uses PyMuPDF to identify and extract all figures from the PDF
2. **Match with Existing References**: Attempts to match extracted figures with existing `ref_entries` in the JSON
3. **Create New References**: For figures without matching references, creates new entries with the prefix `FIGREF_EXTRACTED_`
4. **LLM Analysis**: Sends each figure to the specified vision model for detailed analysis
5. **JSON Enhancement**: Adds figure paths and LLM descriptions to the JSON representation

## Integration with Paper2Code

This module is designed to seamlessly integrate with the existing Paper2Code pipeline:

1. Run after preprocessing the paper with s2orc-doc2json and `0_pdf_process.py`
2. Use the enhanced JSON output in subsequent steps (planning, analysis, coding)
3. All downstream models will benefit from the detailed figure descriptions

## Cost Optimization

The module is optimized for cost-effectiveness:
- Uses prompt caching for efficient token usage
- Only sends each figure for analysis once
- Correctly specifies image formats for optimal processing

## Dependencies

- PyMuPDF (fitz): For PDF figure extraction
- OpenAI API: For vision model access
- Standard libraries: os, json, base64, pathlib

## Example Output

The enhanced JSON will contain entries like:

```json
"ref_entries": {
  "FIGREF3": {
    "type": "figure",
    "text": "Original figure caption...",
    "fig_num": "3",
    "page": 7,
    "image_path": "./figures/figure3.png",
    "llm_caption": "Detailed technical analysis of the figure by the LLM..."
  }
}
```