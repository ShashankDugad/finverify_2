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

# Tokenizer (cl100k_base used by BGE/OpenAI models)
enc = tiktoken.get_encoding("cl100k_base")

def chunk_text(text, chunk_size=512, overlap=50):
    """Split text into overlapping token chunks."""
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += chunk_size - overlap
    return chunks

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Read metadata
    with open(METADATA_CSV, "r") as f:
        reader = csv.DictReader(f)
        files = list(reader)
    
    all_chunks = []
    chunk_id = 0
    
    for row in files:
        filepath = row["filepath"]
        
        # Read transcript
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        
        # Chunk
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
                "text": chunk
            })
            chunk_id += 1
    
    # Save as JSONL
    output_file = os.path.join(OUTPUT_DIR, "chunks.jsonl")
    with open(output_file, "w") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")
    
    print(f"Total transcripts: {len(files)}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()
