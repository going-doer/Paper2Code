#!/usr/bin/env python3
"""
Extract figures from PDFs and get LLM descriptions for them.
This enhances the JSON output with image paths and LLM-generated descriptions.
"""

import os
import json
import base64
import argparse
from pathlib import Path
import fitz  # PyMuPDF
from openai import OpenAI
import shutil

def extract_figures(pdf_path, output_dir):
    """Extract figures from a PDF file using PyMuPDF."""
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    print(f"Extracting figures from {pdf_path} to {figures_dir}")
    
    pdf = fitz.open(pdf_path)
    figure_info = []
    
    for page_num in range(pdf.page_count):
        page = pdf[page_num]
        image_list = page.get_images(full=True)
        
        for img_idx, img in enumerate(image_list):
            xref = img[0]
            
            try:
                base_image = pdf.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Generate a filename for the image
                fig_num = len(figure_info) + 1
                image_filename = f"figure{fig_num}.{image_ext}"
                image_path = os.path.join(figures_dir, image_filename)
                
                # Save the image
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                # Record information about the figure
                figure_info.append({
                    "fig_num": fig_num,
                    "page": page_num + 1,
                    "image_path": f"./figures/{image_filename}",
                    "xref": xref,
                    "ext": image_ext  # Store the extension for later use
                })
                
                print(f"Extracted figure {fig_num} from page {page_num + 1} with extension .{image_ext}")
                
            except Exception as e:
                print(f"Error extracting image at page {page_num + 1}, index {img_idx}: {e}")
    
    return figure_info

def get_llm_descriptions(figure_info, json_data, paper_title, output_dir, gpt_version="o3-mini"):
    """Get LLM descriptions for figures using the OpenAI API."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    print(f"Getting LLM descriptions using {gpt_version}")
    
    # Create ref_entries if it doesn't exist
    if "ref_entries" not in json_data:
        json_data["ref_entries"] = {}
        
    ref_entries = json_data["ref_entries"]
    
    # First attempt to match extracted figures with existing ref_entries
    for fig in figure_info:
        matched = False
        for ref_id, ref_entry in ref_entries.items():
            if ref_entry.get("type") == "figure" and ref_entry.get("page") == fig["page"]:
                # Found a potential match
                matched = True
                image_path = fig["image_path"]
                
                # Read the image and encode it as base64
                with open(os.path.join(output_dir, "figures", f"figure{fig['fig_num']}.{fig['ext']}"), "rb") as img_file:
                    image_bytes = img_file.read()
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                
                # Generate a prompt for the LLM
                prompt = f"""
This is Figure {ref_entry.get('fig_num', 'N/A')} from the scientific paper "{paper_title}".
The figure caption is: "{ref_entry.get('text', 'No caption available')}"

Analyze this figure and provide a detailed technical description of what it shows.
Your analysis should:
1. Explain the visual data, patterns, or processes depicted
2. Connect the figure to the main concepts in the paper
3. Interpret any technical details such as graphs, plots, or diagrams
4. Highlight key findings or insights that this figure demonstrates
5. Avoid just repeating the caption - provide deeper analysis

Your description should be scientific, precise, and include technical details that would be relevant for code implementation.
"""
                
                try:
                    # Get the LLM description using the Vision API
                    response = client.chat.completions.create(
                        model=gpt_version,
                        messages=[
                            {"role": "system", "content": "You are a scientific assistant that provides detailed descriptions of figures from academic papers."},
                            {"role": "user", "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/{fig['ext']};base64,{image_base64}"}}
                            ]}
                        ]
                    )
                    
                    llm_caption = response.choices[0].message.content
                    
                    # Update the ref_entry with the image path and LLM caption
                    ref_entry["image_path"] = image_path
                    ref_entry["llm_caption"] = llm_caption
                    
                    print(f"Added LLM description for {ref_id} (Figure {ref_entry.get('fig_num', 'N/A')})")
                    break
                    
                except Exception as e:
                    print(f"Error getting LLM description: {e}")
        
        # If no matching ref_entry was found, create a new one
        if not matched:
            print(f"Creating new ref_entry for figure {fig['fig_num']} on page {fig['page']}")
            
            image_path = fig["image_path"]
            
            # Read the image and encode it as base64
            with open(os.path.join(output_dir, "figures", f"figure{fig['fig_num']}.{fig['ext']}"), "rb") as img_file:
                image_bytes = img_file.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            
            # Generate a prompt for the LLM to analyze the image
            prompt = f"""
This is a figure from page {fig['page']} of the scientific paper "{paper_title}".
No caption was available in the parsed document.

Analyze this figure and provide a detailed technical description of what it shows.
Your analysis should:
1. Explain the visual data, patterns, or processes depicted in detail
2. Identify the figure type (graph, diagram, chart, illustration, etc.)
3. Interpret any technical details such as axes, scales, or relationships shown
4. Make connections to potential scientific concepts based on what you see
5. Describe how this figure might contribute to the paper's methodology or findings

Your description should be scientific, precise, and include technical details that would be relevant for code implementation.
"""
            
            try:
                # Get the LLM description using the Vision API
                response = client.chat.completions.create(
                    model=gpt_version,
                    messages=[
                        {"role": "system", "content": "You are a scientific assistant that provides detailed descriptions of figures from academic papers."},
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/{fig['ext']};base64,{image_base64}"}}
                        ]}
                    ]
                )
                
                llm_caption = response.choices[0].message.content
                
                # Create a new ref_entry
                ref_id = f"FIGREF_EXTRACTED_{fig['fig_num']}"
                ref_entries[ref_id] = {
                    "type": "figure",
                    "text": "No caption available in the parsed document",
                    "fig_num": str(fig['fig_num']),
                    "page": fig['page'],
                    "image_path": image_path,
                    "llm_caption": llm_caption,
                    "extracted_by_script": True
                }
                
                print(f"Created new ref_entry {ref_id} with LLM description")
                
            except Exception as e:
                print(f"Error getting LLM description: {e}")
    
    return json_data

def enhance_json_with_figures(pdf_path, json_path, output_dir, gpt_version="o3-mini"):
    """Enhance the JSON output with figure information and LLM descriptions."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract the paper title from the JSON
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    paper_title = json_data.get("title", "Unknown Paper")
    
    # Extract figures from the PDF
    figure_info = extract_figures(pdf_path, output_dir)
    
    # Get LLM descriptions for the figures
    enhanced_json = get_llm_descriptions(figure_info, json_data, paper_title, output_dir, gpt_version)
    
    # Save the enhanced JSON
    enhanced_json_path = os.path.join(output_dir, "enhanced_paper.json")
    with open(enhanced_json_path, 'w') as f:
        json.dump(enhanced_json, f, indent=2)
    
    print(f"Enhanced JSON saved to {enhanced_json_path}")
    
    return enhanced_json_path

def main():
    parser = argparse.ArgumentParser(description="Enhance JSON with figures and LLM descriptions")
    parser.add_argument("--pdf_path", type=str, required=True, help="Path to the PDF file")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--gpt_version", type=str, default="o3-mini", help="GPT version to use")
    
    args = parser.parse_args()
    
    enhance_json_with_figures(args.pdf_path, args.json_path, args.output_dir, args.gpt_version)

if __name__ == "__main__":
    main()