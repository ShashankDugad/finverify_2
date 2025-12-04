import json
import numpy as np
from numpy.linalg import norm

EMBEDDINGS_FILE = "/scratch/sd5957/finverify_2/data/processed/embeddings.npy"
METADATA_FILE = "/scratch/sd5957/finverify_2/data/processed/chunk_metadata.jsonl"
CHUNKS_FILE = "/scratch/sd5957/finverify_2/data/processed/chunks/chunks.jsonl"

# Load data
print("Loading embeddings...")
embeddings = np.load(EMBEDDINGS_FILE)

print("Loading metadata...")
metadata = []
with open(METADATA_FILE, "r") as f:
    for line in f:
        metadata.append(json.loads(line))

print("Loading chunks (for text)...")
chunks = []
with open(CHUNKS_FILE, "r") as f:
    for line in f:
        chunks.append(json.loads(line))

print("=" * 60)
print("EMBEDDINGS QUALITY REPORT")
print("=" * 60)

# 1. Basic shape verification
print(f"\n1. BASIC VERIFICATION")
print(f"   Embeddings shape: {embeddings.shape}")
print(f"   Metadata count: {len(metadata)}")
print(f"   Chunks count: {len(chunks)}")
print(f"   Shape match: {embeddings.shape[0] == len(metadata) == len(chunks)}")

# 2. Normalization check (BGE should be L2 normalized)
print(f"\n2. NORMALIZATION CHECK")
norms = np.linalg.norm(embeddings, axis=1)
print(f"   Min norm: {norms.min():.6f}")
print(f"   Max norm: {norms.max():.6f}")
print(f"   Mean norm: {norms.mean():.6f}")
print(f"   All normalized (norm ≈ 1.0): {np.allclose(norms, 1.0, atol=0.01)}")

# 3. Distribution check (no degenerate/zero vectors)
print(f"\n3. DISTRIBUTION CHECK")
print(f"   Dtype: {embeddings.dtype}")
print(f"   Min value: {embeddings.min():.6f}")
print(f"   Max value: {embeddings.max():.6f}")
print(f"   Mean value: {embeddings.mean():.6f}")
print(f"   Std dev: {embeddings.std():.6f}")
zero_vectors = np.sum(np.all(embeddings == 0, axis=1))
print(f"   Zero vectors: {zero_vectors}")

# 4. Variance per dimension (check for dead dimensions)
dim_variance = np.var(embeddings, axis=0)
dead_dims = np.sum(dim_variance < 1e-6)
print(f"\n4. DIMENSION HEALTH")
print(f"   Dead dimensions (var < 1e-6): {dead_dims}")
print(f"   Min dim variance: {dim_variance.min():.6f}")
print(f"   Max dim variance: {dim_variance.max():.6f}")

# 5. Semantic similarity test - same company chunks should be more similar
print(f"\n5. SEMANTIC SIMILARITY TEST")

def cosine_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# Find Apple chunks
apple_indices = [i for i, m in enumerate(metadata) if m["company"] == "Apple"]
nvidia_indices = [i for i, m in enumerate(metadata) if m["company"] == "Nvidia"]
walmart_indices = [i for i, m in enumerate(metadata) if m["company"] == "Walmart"]

# Same company similarity (Apple vs Apple)
if len(apple_indices) >= 2:
    same_company_sims = []
    for i in range(min(10, len(apple_indices)-1)):
        sim = cosine_sim(embeddings[apple_indices[i]], embeddings[apple_indices[i+1]])
        same_company_sims.append(sim)
    print(f"   Apple vs Apple (consecutive chunks): {np.mean(same_company_sims):.4f}")

# Different company similarity (Apple vs Nvidia)
if apple_indices and nvidia_indices:
    diff_company_sims = []
    for i in range(min(10, len(apple_indices), len(nvidia_indices))):
        sim = cosine_sim(embeddings[apple_indices[i]], embeddings[nvidia_indices[i]])
        diff_company_sims.append(sim)
    print(f"   Apple vs Nvidia (different companies): {np.mean(diff_company_sims):.4f}")

# 6. Retrieval test with sample query
print(f"\n6. RETRIEVAL TEST")

# Simulate query embedding (use a chunk about revenue as proxy)
# Find a chunk mentioning "revenue" 
revenue_chunk_idx = None
for i, c in enumerate(chunks):
    if "revenue" in c["text"].lower() and "quarter" in c["text"].lower():
        revenue_chunk_idx = i
        break

if revenue_chunk_idx:
    query_emb = embeddings[revenue_chunk_idx]
    
    # Compute similarities to all chunks
    similarities = np.dot(embeddings, query_emb)
    top_indices = np.argsort(similarities)[::-1][:5]
    
    print(f"   Query chunk ({metadata[revenue_chunk_idx]['company']} {metadata[revenue_chunk_idx]['year']} Q{metadata[revenue_chunk_idx]['quarter']}):")
    print(f"   '{chunks[revenue_chunk_idx]['text'][:100]}...'")
    print(f"\n   Top 5 similar chunks:")
    for rank, idx in enumerate(top_indices):
        m = metadata[idx]
        sim = similarities[idx]
        print(f"   {rank+1}. [{m['company']} {m['year']} Q{m['quarter']}] sim={sim:.4f}")

# 7. Embedding alignment check
print(f"\n7. ALIGNMENT CHECK")
# Verify chunk_id matches between metadata and chunks
mismatches = 0
for i in range(min(100, len(metadata))):
    if metadata[i]["chunk_id"] != chunks[i]["chunk_id"]:
        mismatches += 1
print(f"   Checked first 100 entries")
print(f"   Mismatches: {mismatches}")

print("\n" + "=" * 60)
print("QUALITY ASSESSMENT COMPLETE")
print("=" * 60)
