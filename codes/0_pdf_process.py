import json
import argparse

def remove_spans(data):
    # If data is a dictionary, recursively check its keys
    if isinstance(data, dict):
        # Remove specific keys if present
        for key in ["cite_spans", "ref_spans", "eq_spans", "authors", "bib_entries", \
                    "year", "venue", "identifiers", "_pdf_hash", "header"]:
            data.pop(key, None)
        # Recursively apply to child dictionaries or lists
        for key, value in data.items():
            data[key] = remove_spans(value)
    # If data is a list, apply the function to each item
    elif isinstance(data, list):
        return [remove_spans(item) for item in data]
    return data

def main(args):
    input_json_path = args.input_json_path
    output_json_path = args.output_json_path 

    with open(f'{input_json_path}', encoding='utf-8') as f:
        data = json.load(f)

    cleaned_data = remove_spans(data)

    print(f"[SAVED] {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json_path", type=str)
    parser.add_argument("--output_json_path", type=str)

    
    args = parser.parse_args()
    main(args)

# run
# cd ./s2orc-doc2json/grobid-0.7.3
# ./gradlew run

# python doc2json/grobid2json/process_pdf.py -i tests/pdf/transformer.pdf -t temp_dir/ -o output_dir/