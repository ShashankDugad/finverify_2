import numpy as np
import faiss
import time

EMBEDDINGS_FILE = "/scratch/sd5957/finverify_2/data/processed/embeddings.npy"
INDEX_FILE = "/scratch/sd5957/finverify_2/data/indexes/faiss_index.bin"

# Load embeddings
print("Loading embeddings...")
embeddings = np.load(EMBEDDINGS_FILE).astype('float32')
print(f"Shape: {embeddings.shape}")

n_vectors, dim = embeddings.shape

# Build FAISS index (IndexFlatIP for cosine similarity with normalized vectors)
print("Building FAISS index (IndexFlatIP)...")
start = time.time()

index = faiss.IndexFlatIP(dim)  # Inner product = cosine for normalized vectors
index.add(embeddings)

print(f"Index built in {time.time() - start:.2f}s")
print(f"Total vectors in index: {index.ntotal}")

# Save index
faiss.write_index(index, INDEX_FILE)
print(f"Saved to: {INDEX_FILE}")

# Test retrieval
print("\nTest retrieval (query = first vector):")
query = embeddings[0:1]
D, I = index.search(query, 5)
print(f"Top 5 indices: {I[0]}")
print(f"Top 5 scores: {D[0]}")

print("\nDone!")
