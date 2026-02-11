# process_dataset.py
import os
import pandas as pd
from tqdm import tqdm
from preprocessing import clean_text

BASE = os.path.join(os.path.dirname(__file__), "..")
INPUT = os.path.join(BASE, "data", "raw_sample.csv")
OUTPUT = os.path.join(BASE, "data", "processed_sample.csv")

def process_dataset(input_path=INPUT, output_path=OUTPUT, chunksize=2000):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found")
    processed_chunks = []
    reader = pd.read_csv(input_path, chunksize=chunksize, on_bad_lines='warn', dtype=str)
    for chunk in tqdm(reader, desc="Processing"):
        if 'raw' not in chunk.columns:
            # try to detect alternate column names
            if 0 in chunk.columns:
                chunk.columns = ['raw'] + list(chunk.columns[1:])
            else:
                raise ValueError("'raw' column not found in input CSV")
        chunk['raw'] = chunk['raw'].astype(str).str.strip().str.strip('"').str.strip("'")
        chunk['processed'] = chunk['raw'].apply(clean_text)
        processed_chunks.append(chunk[['raw', 'processed']])
    if processed_chunks:
        final_df = pd.concat(processed_chunks, ignore_index=True)
        final_df.to_csv(output_path, index=False)
        print("Processed file saved to:", output_path)
        print("Total rows in final file:", len(final_df))
    else:
        print("No data processed.")

if __name__ == "__main__":
    process_dataset()
