import json
import pickle
import re
from rank_bm25 import BM25Okapi
import time

CHUNKS_FILE = "/scratch/sd5957/finverify_2/data/processed/chunks/chunks.jsonl"
INDEX_FILE = "/scratch/sd5957/finverify_2/data/indexes/bm25_index.pkl"

def tokenize(text):
    """Improved tokenization: lowercase, remove punctuation, split on whitespace."""
    # Lowercase
    text = text.lower()
    # Remove punctuation (keep alphanumeric and spaces)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Split and remove empty tokens
    tokens = [t for t in text.split() if len(t) > 1]  # Remove single chars
    return tokens

# Load chunks
print("Loading chunks...")
chunks = []
with open(CHUNKS_FILE, "r") as f:
    for line in f:
        chunks.append(json.loads(line))

texts = [c["text"] for c in chunks]
print(f"Total chunks: {len(texts)}")

# Tokenize with improved method
print("Tokenizing (improved)...")
tokenized = [tokenize(text) for text in texts]

# Show sample tokenization
print(f"\nSample tokenization:")
sample = texts[0][:100]
sample_tokens = tokenize(sample)
print(f"   Original: '{sample}...'")
print(f"   Tokens: {sample_tokens[:15]}")

# Build BM25 index
print("\nBuilding BM25 index...")
start = time.time()
bm25 = BM25Okapi(tokenized)
print(f"Index built in {time.time() - start:.2f}s")

# Save index with tokenizer function
print("Saving index...")
with open(INDEX_FILE, "wb") as f:
    pickle.dump({"bm25": bm25, "tokenized": tokenized}, f)
print(f"Saved to: {INDEX_FILE}")

# Test retrieval
print("\nTest retrieval (query = 'quarterly revenue growth'):")
query = "quarterly revenue growth"
query_tokens = tokenize(query)
print(f"   Query tokens: {query_tokens}")
scores = bm25.get_scores(query_tokens)
top_indices = scores.argsort()[::-1][:5]
print(f"   Top 5 indices: {top_indices}")
print(f"   Top 5 scores: {scores[top_indices]}")

# Test another query
print("\nTest retrieval (query = 'Apple's iPhone sales Q4 2024'):")
query2 = "Apple's iPhone sales Q4 2024"
query_tokens2 = tokenize(query2)
print(f"   Query tokens: {query_tokens2}")
scores2 = bm25.get_scores(query_tokens2)
top_indices2 = scores2.argsort()[::-1][:5]
print(f"   Top 5 indices: {top_indices2}")
print(f"   Top 5 scores: {scores2[top_indices2]}")

print("\nDone!")
