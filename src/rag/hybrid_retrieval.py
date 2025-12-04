import json
import numpy as np
import pickle
import faiss
import re
from sentence_transformers import SentenceTransformer
from collections import defaultdict

# Paths
FAISS_INDEX = "/scratch/sd5957/finverify_2/data/indexes/faiss_index.bin"
BM25_INDEX = "/scratch/sd5957/finverify_2/data/indexes/bm25_index.pkl"
CHUNKS_FILE = "/scratch/sd5957/finverify_2/data/processed/chunks/chunks.jsonl"
METADATA_FILE = "/scratch/sd5957/finverify_2/data/processed/chunk_metadata.jsonl"

class HybridRetriever:
    def __init__(self, k=60):
        """
        k: RRF constant (default 60, standard in literature)
        """
        self.k = k
        
        print("Loading FAISS index...")
        self.faiss_index = faiss.read_index(FAISS_INDEX)
        
        print("Loading BM25 index...")
        with open(BM25_INDEX, "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25 = bm25_data["bm25"]
        
        print("Loading chunks and metadata...")
        self.chunks = []
        with open(CHUNKS_FILE, "r") as f:
            for line in f:
                self.chunks.append(json.loads(line))
        
        self.metadata = []
        with open(METADATA_FILE, "r") as f:
            for line in f:
                self.metadata.append(json.loads(line))
        
        print("Loading BGE model...")
        self.model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")
        
        print("Hybrid retriever ready!")
    
    def tokenize(self, text):
        """BM25 tokenization."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = [t for t in text.split() if len(t) > 1]
        return tokens
    
    def search_faiss(self, query, top_k=100):
        """Semantic search with FAISS."""
        query_emb = self.model.encode([query], normalize_embeddings=True).astype('float32')
        scores, indices = self.faiss_index.search(query_emb, top_k)
        return list(zip(indices[0], scores[0]))
    
    def search_bm25(self, query, top_k=100):
        """Lexical search with BM25."""
        tokens = self.tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = scores.argsort()[::-1][:top_k]
        return [(idx, scores[idx]) for idx in top_indices]
    
    def reciprocal_rank_fusion(self, result_lists, k=None):
        """
        Combine multiple ranked lists using RRF.
        
        RRF score = sum(1 / (k + rank_i)) for each list
        
        Args:
            result_lists: list of [(doc_id, score), ...] from each retriever
            k: constant (default 60)
        
        Returns:
            Sorted list of (doc_id, rrf_score)
        """
        if k is None:
            k = self.k
        
        rrf_scores = defaultdict(float)
        
        for results in result_lists:
            for rank, (doc_id, _) in enumerate(results):
                rrf_scores[doc_id] += 1.0 / (k + rank + 1)  # rank is 0-indexed
        
        # Sort by RRF score descending
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results
    
    def search_hybrid(self, query, top_k=10, faiss_weight=1.0, bm25_weight=1.0):
        """
        Hybrid search combining FAISS and BM25 with RRF.
        
        Args:
            query: search query
            top_k: number of results to return
            faiss_weight: weight for FAISS results (for weighted RRF)
            bm25_weight: weight for BM25 results (for weighted RRF)
        
        Returns:
            List of (doc_id, rrf_score, metadata, text)
        """
        # Get results from both retrievers
        faiss_results = self.search_faiss(query, top_k=100)
        bm25_results = self.search_bm25(query, top_k=100)
        
        # Apply RRF
        rrf_results = self.reciprocal_rank_fusion([faiss_results, bm25_results])
        
        # Get top_k results with metadata
        results = []
        for doc_id, rrf_score in rrf_results[:top_k]:
            results.append({
                "doc_id": doc_id,
                "rrf_score": rrf_score,
                "metadata": self.metadata[doc_id],
                "text": self.chunks[doc_id]["text"]
            })
        
        return results
    
    def search_faiss_only(self, query, top_k=10):
        """FAISS-only search for comparison."""
        results = self.search_faiss(query, top_k)
        return [{
            "doc_id": doc_id,
            "score": score,
            "metadata": self.metadata[doc_id],
            "text": self.chunks[doc_id]["text"]
        } for doc_id, score in results]
    
    def search_bm25_only(self, query, top_k=10):
        """BM25-only search for comparison."""
        results = self.search_bm25(query, top_k)
        return [{
            "doc_id": doc_id,
            "score": score,
            "metadata": self.metadata[doc_id],
            "text": self.chunks[doc_id]["text"]
        } for doc_id, score in results]


def test_retriever():
    """Test the hybrid retriever."""
    retriever = HybridRetriever()
    
    test_queries = [
        "What was Apple's iPhone revenue in Q4 2023?",
        "NVIDIA data center GPU sales growth",
        "Microsoft Azure cloud computing revenue",
        "Amazon AWS operating profit margin"
    ]
    
    print("\n" + "=" * 70)
    print("HYBRID RETRIEVAL TEST (RRF)")
    print("=" * 70)
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: '{query}'")
        print("=" * 70)
        
        # Hybrid results
        hybrid_results = retriever.search_hybrid(query, top_k=5)
        print(f"\n[HYBRID (RRF)] Top 5:")
        for i, r in enumerate(hybrid_results):
            m = r["metadata"]
            print(f"   {i+1}. [{m['company']} {m['year']} Q{m['quarter']}] RRF={r['rrf_score']:.4f}")
        
        # FAISS-only results
        faiss_results = retriever.search_faiss_only(query, top_k=5)
        print(f"\n[FAISS Only] Top 5:")
        for i, r in enumerate(faiss_results):
            m = r["metadata"]
            print(f"   {i+1}. [{m['company']} {m['year']} Q{m['quarter']}] score={r['score']:.4f}")
        
        # BM25-only results
        bm25_results = retriever.search_bm25_only(query, top_k=5)
        print(f"\n[BM25 Only] Top 5:")
        for i, r in enumerate(bm25_results):
            m = r["metadata"]
            print(f"   {i+1}. [{m['company']} {m['year']} Q{m['quarter']}] score={r['score']:.4f}")
        
        # Show overlap
        hybrid_ids = set(r["doc_id"] for r in hybrid_results)
        faiss_ids = set(r["doc_id"] for r in faiss_results)
        bm25_ids = set(r["doc_id"] for r in bm25_results)
        
        print(f"\n   Overlap: FAISS∩BM25={len(faiss_ids & bm25_ids)}, "
              f"Hybrid∩FAISS={len(hybrid_ids & faiss_ids)}, "
              f"Hybrid∩BM25={len(hybrid_ids & bm25_ids)}")


if __name__ == "__main__":
    test_retriever()
