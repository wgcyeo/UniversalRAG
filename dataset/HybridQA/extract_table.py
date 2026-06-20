import os
import json
import pandas as pd
from tqdm import tqdm

def format_table_as_text(table_data):
    text_lines = []
    
    if 'title' in table_data:
        text_lines.append(f"Table: {table_data['title']}")
        text_lines.append("")
    
    if 'header' in table_data and table_data['header']:
        header_row = " | ".join([cell[0] if isinstance(cell, list) and len(cell) > 0 else str(cell) 
                                for cell in table_data['header']])
        text_lines.append(header_row)
        text_lines.append("-" * len(header_row))
    
    if 'data' in table_data and table_data['data']:
        for row in table_data['data']:
            row_text = " | ".join([cell[0] if isinstance(cell, list) and len(cell) > 0 else str(cell) 
                                  for cell in row])
            text_lines.append(row_text)
    
    return "\n".join(text_lines)

def extract_table(query_file, tables_tok_path, output_file):
    with open(query_file, 'r') as f:
        hybridqa_data = json.load(f)

    print(f"Processing {len(hybridqa_data)} entries from {query_file}...")

    rows = []
    seen_ids = set()

    for entry in tqdm(hybridqa_data, desc="Extracting tables"):
        gt_table = entry['gt_tables']
        table_file_path = os.path.join(tables_tok_path, f"{gt_table}.json")
        
        if os.path.exists(table_file_path):
            try:
                with open(table_file_path, 'r') as f:
                    table_data = json.load(f)
                
                # Format table as text
                table_text = format_table_as_text(table_data)
                
                psg_id = f"hybridqa-{gt_table}"
                if psg_id not in seen_ids:
                    seen_ids.add(psg_id)
                    rows.append({
                        'psg_id': psg_id,
                        'text': table_text
                    })
            
            except Exception as e:
                print(f"Error processing {table_file_path}: {e}")
        else:
            print(f"Warning: Table file not found: {table_file_path}")
        
    df = pd.DataFrame(rows)
    df.set_index('psg_id', inplace=True)
    df.to_parquet(output_file)
    print(f"Saved {len(df)} tables to {output_file}")

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Extract tables from HybridQA tables_tok files.")
    parser.add_argument('--query', type=str, default='../query/hybridqa.json', help='Query JSON file')
    parser.add_argument('--tables-tok', type=str, default='WikiTables-WithLinks/tables_tok', help='Input directory for tables_tok files')
    parser.add_argument('--output', type=str, default='hybridqa_table.parquet', help='Output parquet file for extracted tables')
    args = parser.parse_args()

    query_file = args.query
    tables_tok_path = args.tables_tok
    output_file = args.output

    extract_table(query_file, tables_tok_path, output_file)