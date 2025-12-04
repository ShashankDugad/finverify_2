import os
import csv
import json
import tiktoken
from pathlib import Path

# Config
METADATA_CSV = "/scratch/sd5957/finverify_2/data/processed/metadata.csv"
OUTPUT_DIR = "/scratch/sd5957/finverify_2/data/processed/chunks"
CHUNK_SIZE = 512
OVERLAP = 50

# Tokenizer
enc = tiktoken.get_encoding("cl100k_base")

def chunk_text(text, chunk_size=512, overlap=50):
    """Split text into overlapping token chunks."""
    tokens = enc.encode(text)
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append({
            "text": chunk_text,
            "start_token": start,
            "end_token": end,
            "token_count": len(chunk_tokens)
        })
        
        # Move forward by (chunk_size - overlap), but ensure progress
        step = chunk_size - overlap
        if start + step >= len(tokens):
            break
        start += step
    
    return chunks

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(METADATA_CSV, "r") as f:
        reader = csv.DictReader(f)
        files = list(reader)
    
    all_chunks = []
    chunk_id = 0
    
    for row in files:
        filepath = row["filepath"]
        
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        
        chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)
        
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": chunk_id,
                "company": row["company"],
                "ticker": row["ticker"],
                "year": row["year"],
                "quarter": row["quarter"],
                "chunk_index": i,
                "total_chunks": len(chunks),
                "token_count": chunk["token_count"],
                "text": chunk["text"]
            })
            chunk_id += 1
    
    # Save as JSONL
    output_file = os.path.join(OUTPUT_DIR, "chunks_v2.jsonl")
    with open(output_file, "w") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")
    
    print(f"Total transcripts: {len(files)}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()
