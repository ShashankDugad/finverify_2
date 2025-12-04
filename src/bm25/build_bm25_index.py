import json
import pickle
from rank_bm25 import BM25Okapi
import time

CHUNKS_FILE = "/scratch/sd5957/finverify_2/data/processed/chunks/chunks.jsonl"
INDEX_FILE = "/scratch/sd5957/finverify_2/data/indexes/bm25_index.pkl"

# Load chunks
print("Loading chunks...")
chunks = []
with open(CHUNKS_FILE, "r") as f:
    for line in f:
        chunks.append(json.loads(line))

texts = [c["text"] for c in chunks]
print(f"Total chunks: {len(texts)}")

# Tokenize (simple whitespace + lowercase)
print("Tokenizing...")
tokenized = [text.lower().split() for text in texts]

# Build BM25 index
print("Building BM25 index...")
start = time.time()
bm25 = BM25Okapi(tokenized)
print(f"Index built in {time.time() - start:.2f}s")

# Save index
print("Saving index...")
with open(INDEX_FILE, "wb") as f:
    pickle.dump({"bm25": bm25, "tokenized": tokenized}, f)
print(f"Saved to: {INDEX_FILE}")

# Test retrieval
print("\nTest retrieval (query = 'quarterly revenue growth'):")
query = "quarterly revenue growth"
query_tokens = query.lower().split()
scores = bm25.get_scores(query_tokens)
top_indices = scores.argsort()[::-1][:5]
print(f"Top 5 indices: {top_indices}")
print(f"Top 5 scores: {scores[top_indices]}")

print("\nDone!")
