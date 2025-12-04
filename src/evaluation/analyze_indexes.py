import json
import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# Paths
FAISS_INDEX = "/scratch/sd5957/finverify_2/data/indexes/faiss_index.bin"
BM25_INDEX = "/scratch/sd5957/finverify_2/data/indexes/bm25_index.pkl"
CHUNKS_FILE = "/scratch/sd5957/finverify_2/data/processed/chunks/chunks.jsonl"
METADATA_FILE = "/scratch/sd5957/finverify_2/data/processed/chunk_metadata.jsonl"
EMBEDDINGS_FILE = "/scratch/sd5957/finverify_2/data/processed/embeddings.npy"

print("Loading data...")
# Load FAISS
faiss_index = faiss.read_index(FAISS_INDEX)

# Load BM25
with open(BM25_INDEX, "rb") as f:
    bm25_data = pickle.load(f)
bm25 = bm25_data["bm25"]

# Load chunks and metadata
chunks = []
with open(CHUNKS_FILE, "r") as f:
    for line in f:
        chunks.append(json.loads(line))

metadata = []
with open(METADATA_FILE, "r") as f:
    for line in f:
        metadata.append(json.loads(line))

# Load embeddings for verification
embeddings = np.load(EMBEDDINGS_FILE)

# Load model for query encoding
print("Loading BGE model...")
model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")

print("=" * 70)
print("INDEX QUALITY REPORT")
print("=" * 70)

# 1. Basic integrity checks
print(f"\n1. INTEGRITY CHECK")
print(f"   FAISS index vectors: {faiss_index.ntotal}")
print(f"   BM25 corpus size: {bm25.corpus_size}")
print(f"   Chunks count: {len(chunks)}")
print(f"   Metadata count: {len(metadata)}")
print(f"   Embeddings shape: {embeddings.shape}")
all_match = faiss_index.ntotal == len(chunks) == len(metadata) == embeddings.shape[0]
print(f"   All counts match: {all_match}")

# 2. FAISS index verification
print(f"\n2. FAISS INDEX VERIFICATION")
test_indices = [0, 100, 1000, 10000, 28000]
print(f"   Testing vector reconstruction at indices: {test_indices}")
for idx in test_indices:
    query = embeddings[idx:idx+1].astype('float32')
    D, I = faiss_index.search(query, 1)
    match = I[0][0] == idx
    print(f"   Index {idx}: self-retrieval = {match}, score = {D[0][0]:.6f}")

# 3. Test queries for financial domain
print(f"\n3. RETRIEVAL QUALITY TEST")

test_queries = [
    "What was Apple's quarterly revenue?",
    "NVIDIA GPU sales and data center growth",
    "Microsoft cloud Azure revenue increase",
    "Amazon AWS profit margin",
    "Tesla electric vehicle deliveries"
]

def search_faiss(query_text, top_k=5):
    query_emb = model.encode([query_text], normalize_embeddings=True).astype('float32')
    D, I = faiss_index.search(query_emb, top_k)
    return I[0], D[0]

def search_bm25(query_text, top_k=5):
    tokens = query_text.lower().split()
    scores = bm25.get_scores(tokens)
    top_indices = scores.argsort()[::-1][:top_k]
    return top_indices, scores[top_indices]

for query in test_queries:
    print(f"\n   Query: '{query}'")
    
    # FAISS results
    faiss_ids, faiss_scores = search_faiss(query)
    print(f"   FAISS Top 3:")
    for i, (idx, score) in enumerate(zip(faiss_ids[:3], faiss_scores[:3])):
        m = metadata[idx]
        print(f"      {i+1}. [{m['company']} {m['year']} Q{m['quarter']}] score={score:.4f}")
    
    # BM25 results
    bm25_ids, bm25_scores = search_bm25(query)
    print(f"   BM25 Top 3:")
    for i, (idx, score) in enumerate(zip(bm25_ids[:3], bm25_scores[:3])):
        m = metadata[idx]
        print(f"      {i+1}. [{m['company']} {m['year']} Q{m['quarter']}] score={score:.4f}")

# 4. Company coverage check
print(f"\n4. COMPANY COVERAGE IN RETRIEVAL")
from collections import Counter
all_retrieved_companies = []

random_indices = np.random.choice(len(chunks), 20, replace=False)
for idx in random_indices:
    query_emb = embeddings[idx:idx+1].astype('float32')
    D, I = faiss_index.search(query_emb, 10)
    for retrieved_idx in I[0]:
        all_retrieved_companies.append(metadata[retrieved_idx]["company"])

company_counts = Counter(all_retrieved_companies)
print(f"   Companies retrieved (from 20 random queries, top 10 each):")
for company, count in company_counts.most_common(10):
    print(f"      {company}: {count}")

# 5. Score distribution check
print(f"\n5. SCORE DISTRIBUTION")
sample_indices = np.random.choice(len(chunks), 100, replace=False)
all_scores = []
for idx in sample_indices:
    query_emb = embeddings[idx:idx+1].astype('float32')
    D, I = faiss_index.search(query_emb, 10)
    all_scores.extend(D[0][1:])  # Exclude self-match

print(f"   FAISS similarity scores (excluding self-match):")
print(f"      Min: {min(all_scores):.4f}")
print(f"      Max: {max(all_scores):.4f}")
print(f"      Mean: {np.mean(all_scores):.4f}")
print(f"      Std: {np.std(all_scores):.4f}")

# 6. BM25 tokenization check
print(f"\n6. BM25 TOKENIZATION CHECK")
sample_chunk = chunks[0]["text"][:200]
tokens = sample_chunk.lower().split()
print(f"   Sample text: '{sample_chunk[:100]}...'")
print(f"   Token count: {len(tokens)}")
print(f"   First 10 tokens: {tokens[:10]}")

# 7. Cross-method agreement
print(f"\n7. CROSS-METHOD AGREEMENT")
agreement_scores = []
for query in test_queries:
    faiss_ids, _ = search_faiss(query, top_k=10)
    bm25_ids, _ = search_bm25(query, top_k=10)
    overlap = len(set(faiss_ids) & set(bm25_ids))
    agreement_scores.append(overlap)
    print(f"   '{query[:40]}...': {overlap}/10 overlap")

print(f"\n   Average FAISS-BM25 overlap: {np.mean(agreement_scores):.1f}/10")

print("\n" + "=" * 70)
print("INDEX QUALITY ASSESSMENT COMPLETE")
print("=" * 70)
