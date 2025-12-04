import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Config
CHUNKS_FILE = "/scratch/sd5957/finverify_2/data/processed/chunks/chunks.jsonl"
OUTPUT_FILE = "/scratch/sd5957/finverify_2/data/processed/embeddings.npy"
METADATA_FILE = "/scratch/sd5957/finverify_2/data/processed/chunk_metadata.jsonl"
MODEL_NAME = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 32

# Check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model
print(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME, device=device)

# Load chunks
print("Loading chunks...")
chunks = []
with open(CHUNKS_FILE, "r") as f:
    for line in f:
        chunks.append(json.loads(line))

texts = [c["text"] for c in chunks]
print(f"Total chunks: {len(texts)}")

# Generate embeddings in batches
print("Generating embeddings...")
embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True  # For cosine similarity
)

print(f"Embeddings shape: {embeddings.shape}")

# Save embeddings
np.save(OUTPUT_FILE, embeddings)
print(f"Saved embeddings to: {OUTPUT_FILE}")

# Save metadata (without text, for faster loading)
with open(METADATA_FILE, "w") as f:
    for c in chunks:
        meta = {k: v for k, v in c.items() if k != "text"}
        f.write(json.dumps(meta) + "\n")
print(f"Saved metadata to: {METADATA_FILE}")

print("Done!")
