import json
import numpy as np
import pickle
import faiss
import re
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from collections import defaultdict

# Paths
FAISS_INDEX = "/scratch/sd5957/finverify_2/data/indexes/faiss_index.bin"
BM25_INDEX = "/scratch/sd5957/finverify_2/data/indexes/bm25_index.pkl"
CHUNKS_FILE = "/scratch/sd5957/finverify_2/data/processed/chunks/chunks.jsonl"
METADATA_FILE = "/scratch/sd5957/finverify_2/data/processed/chunk_metadata.jsonl"

class RerankedRetriever:
    def __init__(self, k=60):
        """
        Full retrieval pipeline: FAISS + BM25 → RRF → Cross-Encoder Reranking
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
        
        print("Loading BGE model (bi-encoder)...")
        self.bi_encoder = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")
        
        print("Loading Cross-Encoder (reranker)...")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda")
        
        print("Reranked retriever ready!")
    
    def tokenize(self, text):
        """BM25 tokenization."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = [t for t in text.split() if len(t) > 1]
        return tokens
    
    def search_faiss(self, query, top_k=100):
        """Semantic search with FAISS."""
        query_emb = self.bi_encoder.encode([query], normalize_embeddings=True).astype('float32')
        scores, indices = self.faiss_index.search(query_emb, top_k)
        return list(zip(indices[0], scores[0]))
    
    def search_bm25(self, query, top_k=100):
        """Lexical search with BM25."""
        tokens = self.tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = scores.argsort()[::-1][:top_k]
        return [(idx, scores[idx]) for idx in top_indices]
    
    def reciprocal_rank_fusion(self, result_lists):
        """Combine ranked lists using RRF."""
        rrf_scores = defaultdict(float)
        
        for results in result_lists:
            for rank, (doc_id, _) in enumerate(results):
                rrf_scores[doc_id] += 1.0 / (self.k + rank + 1)
        
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results
    
    def rerank(self, query, doc_ids, top_k=10):
        """
        Rerank documents using cross-encoder.
        
        Args:
            query: search query
            doc_ids: list of document IDs to rerank
            top_k: number of results to return
        
        Returns:
            Reranked list of (doc_id, cross_encoder_score)
        """
        # Prepare query-document pairs
        pairs = [(query, self.chunks[doc_id]["text"][:512]) for doc_id in doc_ids]
        
        # Get cross-encoder scores
        scores = self.cross_encoder.predict(pairs)
        
        # Sort by score
        doc_scores = list(zip(doc_ids, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return doc_scores[:top_k]
    
    def search(self, query, initial_top_k=100, final_top_k=10):
        """
        Full pipeline: FAISS + BM25 → RRF → Cross-Encoder Reranking
        
        Args:
            query: search query
            initial_top_k: candidates from each retriever for RRF
            final_top_k: final results after reranking
        
        Returns:
            List of results with metadata
        """
        # Stage 1: Get candidates from FAISS and BM25
        faiss_results = self.search_faiss(query, top_k=initial_top_k)
        bm25_results = self.search_bm25(query, top_k=initial_top_k)
        
        # Stage 2: RRF fusion
        rrf_results = self.reciprocal_rank_fusion([faiss_results, bm25_results])
        
        # Get top candidates for reranking (e.g., top 50)
        candidate_ids = [doc_id for doc_id, _ in rrf_results[:50]]
        
        # Stage 3: Cross-encoder reranking
        reranked = self.rerank(query, candidate_ids, top_k=final_top_k)
        
        # Build final results
        results = []
        for doc_id, ce_score in reranked:
            results.append({
                "doc_id": doc_id,
                "ce_score": float(ce_score),
                "metadata": self.metadata[doc_id],
                "text": self.chunks[doc_id]["text"]
            })
        
        return results
    
    def search_without_reranking(self, query, top_k=10):
        """RRF only (no reranking) for comparison."""
        faiss_results = self.search_faiss(query, top_k=100)
        bm25_results = self.search_bm25(query, top_k=100)
        rrf_results = self.reciprocal_rank_fusion([faiss_results, bm25_results])
        
        results = []
        for doc_id, rrf_score in rrf_results[:top_k]:
            results.append({
                "doc_id": doc_id,
                "rrf_score": rrf_score,
                "metadata": self.metadata[doc_id],
                "text": self.chunks[doc_id]["text"]
            })
        
        return results


def test_reranker():
    """Test the full retrieval pipeline."""
    retriever = RerankedRetriever()
    
    test_queries = [
        "What was Apple's iPhone revenue in Q4 2023?",
        "NVIDIA data center GPU sales growth",
        "Microsoft Azure cloud computing revenue",
        "Amazon AWS operating profit margin"
    ]
    
    print("\n" + "=" * 70)
    print("CROSS-ENCODER RERANKING TEST")
    print("=" * 70)
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: '{query}'")
        print("=" * 70)
        
        # Without reranking (RRF only)
        rrf_results = retriever.search_without_reranking(query, top_k=5)
        print(f"\n[RRF Only] Top 5:")
        for i, r in enumerate(rrf_results):
            m = r["metadata"]
            print(f"   {i+1}. [{m['company']} {m['year']} Q{m['quarter']}] RRF={r['rrf_score']:.4f}")
        
        # With reranking
        reranked_results = retriever.search(query, final_top_k=5)
        print(f"\n[RRF + Cross-Encoder] Top 5:")
        for i, r in enumerate(reranked_results):
            m = r["metadata"]
            print(f"   {i+1}. [{m['company']} {m['year']} Q{m['quarter']}] CE={r['ce_score']:.4f}")
        
        # Show rank changes
        rrf_ids = [r["doc_id"] for r in rrf_results]
        ce_ids = [r["doc_id"] for r in reranked_results]
        
        promoted = set(ce_ids) - set(rrf_ids)
        demoted = set(rrf_ids) - set(ce_ids)
        
        print(f"\n   Rank changes: {len(promoted)} promoted, {len(demoted)} demoted")


if __name__ == "__main__":
    test_reranker()
