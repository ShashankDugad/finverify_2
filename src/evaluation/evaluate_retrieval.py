import json
import numpy as np
import pickle
import faiss
import re
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer, CrossEncoder
from collections import defaultdict
import os

# Paths
FAISS_INDEX = "/scratch/sd5957/finverify_2/data/indexes/faiss_index.bin"
BM25_INDEX = "/scratch/sd5957/finverify_2/data/indexes/bm25_index.pkl"
CHUNKS_FILE = "/scratch/sd5957/finverify_2/data/processed/chunks/chunks.jsonl"
METADATA_FILE = "/scratch/sd5957/finverify_2/data/processed/chunk_metadata.jsonl"
OUTPUT_DIR = "/scratch/sd5957/finverify_2/data/outputs/figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style for poster-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class Evaluator:
    def __init__(self):
        print("Loading indexes...")
        self.faiss_index = faiss.read_index(FAISS_INDEX)
        
        with open(BM25_INDEX, "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25 = bm25_data["bm25"]
        
        self.chunks = []
        with open(CHUNKS_FILE, "r") as f:
            for line in f:
                self.chunks.append(json.loads(line))
        
        self.metadata = []
        with open(METADATA_FILE, "r") as f:
            for line in f:
                self.metadata.append(json.loads(line))
        
        print("Loading models...")
        self.bi_encoder = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda")
        print("Ready!")
    
    def tokenize(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return [t for t in text.split() if len(t) > 1]
    
    def search_faiss(self, query, top_k=100):
        query_emb = self.bi_encoder.encode([query], normalize_embeddings=True).astype('float32')
        scores, indices = self.faiss_index.search(query_emb, top_k)
        return list(zip(indices[0], scores[0]))
    
    def search_bm25(self, query, top_k=100):
        tokens = self.tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = scores.argsort()[::-1][:top_k]
        return [(idx, scores[idx]) for idx in top_indices]
    
    def rrf(self, result_lists, k=60):
        rrf_scores = defaultdict(float)
        for results in result_lists:
            for rank, (doc_id, _) in enumerate(results):
                rrf_scores[doc_id] += 1.0 / (k + rank + 1)
        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    def rerank(self, query, doc_ids, top_k=10):
        pairs = [(query, self.chunks[doc_id]["text"][:512]) for doc_id in doc_ids]
        scores = self.cross_encoder.predict(pairs)
        doc_scores = list(zip(doc_ids, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return doc_scores[:top_k]
    
    def get_results(self, query, method="hybrid_reranked", top_k=10):
        """Get results for different methods."""
        if method == "faiss":
            results = self.search_faiss(query, top_k)
            return [doc_id for doc_id, _ in results]
        
        elif method == "bm25":
            results = self.search_bm25(query, top_k)
            return [doc_id for doc_id, _ in results]
        
        elif method == "rrf":
            faiss_results = self.search_faiss(query, 100)
            bm25_results = self.search_bm25(query, 100)
            rrf_results = self.rrf([faiss_results, bm25_results])
            return [doc_id for doc_id, _ in rrf_results[:top_k]]
        
        elif method == "hybrid_reranked":
            faiss_results = self.search_faiss(query, 100)
            bm25_results = self.search_bm25(query, 100)
            rrf_results = self.rrf([faiss_results, bm25_results])
            candidate_ids = [doc_id for doc_id, _ in rrf_results[:50]]
            reranked = self.rerank(query, candidate_ids, top_k)
            return [doc_id for doc_id, _ in reranked]
    
    def is_relevant(self, doc_id, query_company, query_year=None):
        """Check if document is relevant (same company, optionally same year)."""
        meta = self.metadata[doc_id]
        if meta["company"].lower() != query_company.lower():
            return False
        if query_year and str(meta["year"]) != str(query_year):
            return False
        return True
    
    def compute_metrics(self, retrieved_ids, query_company, query_year=None, k_values=[1, 3, 5, 10]):
        """Compute Precision@K, Recall@K, MRR."""
        metrics = {}
        
        # Count relevant in corpus
        total_relevant = sum(1 for i in range(len(self.metadata)) 
                           if self.is_relevant(i, query_company, query_year))
        
        for k in k_values:
            top_k = retrieved_ids[:k]
            relevant_retrieved = sum(1 for doc_id in top_k 
                                    if self.is_relevant(doc_id, query_company, query_year))
            
            metrics[f"P@{k}"] = relevant_retrieved / k if k > 0 else 0
            metrics[f"R@{k}"] = relevant_retrieved / total_relevant if total_relevant > 0 else 0
        
        # MRR
        for rank, doc_id in enumerate(retrieved_ids):
            if self.is_relevant(doc_id, query_company, query_year):
                metrics["MRR"] = 1.0 / (rank + 1)
                break
        else:
            metrics["MRR"] = 0.0
        
        return metrics


def run_evaluation():
    evaluator = Evaluator()
    
    # Test queries with ground truth company
    test_queries = [
        ("What was Apple's iPhone revenue?", "Apple", None),
        ("NVIDIA data center GPU sales", "Nvidia", None),
        ("Microsoft Azure cloud revenue", "Microsoft", None),
        ("Amazon AWS profit margin", "Amazon", None),
        ("Apple quarterly earnings Q4 2023", "Apple", "2023"),
        ("Google cloud platform growth", "Alphabet", None),
        ("Meta advertising revenue", "META", None),
        ("Walmart e-commerce sales growth", "Walmart", None),
    ]
    
    methods = ["faiss", "bm25", "rrf", "hybrid_reranked"]
    method_names = ["FAISS\n(Semantic)", "BM25\n(Lexical)", "RRF\n(Hybrid)", "RRF +\nReranker"]
    
    # Collect metrics
    all_metrics = {method: [] for method in methods}
    
    print("\n" + "=" * 70)
    print("RETRIEVAL EVALUATION")
    print("=" * 70)
    
    for query, company, year in test_queries:
        print(f"\nQuery: '{query[:50]}...' (Company: {company})")
        
        for method in methods:
            results = evaluator.get_results(query, method, top_k=10)
            metrics = evaluator.compute_metrics(results, company, year)
            all_metrics[method].append(metrics)
            print(f"  {method:18} P@5={metrics['P@5']:.2f} R@10={metrics['R@10']:.2f} MRR={metrics['MRR']:.2f}")
    
    # Average metrics
    print("\n" + "=" * 70)
    print("AVERAGE METRICS")
    print("=" * 70)
    
    avg_metrics = {}
    for method in methods:
        avg_metrics[method] = {}
        for key in all_metrics[method][0].keys():
            avg_metrics[method][key] = np.mean([m[key] for m in all_metrics[method]])
        print(f"{method:18} P@5={avg_metrics[method]['P@5']:.3f} R@10={avg_metrics[method]['R@10']:.3f} MRR={avg_metrics[method]['MRR']:.3f}")
    
    # ==================== VISUALIZATIONS ====================
    print("\n" + "=" * 70)
    print("GENERATING POSTER FIGURES")
    print("=" * 70)
    
    # Figure 1: Bar chart comparing methods
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    metrics_to_plot = ["P@5", "R@10", "MRR"]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    
    for ax, metric in zip(axes, metrics_to_plot):
        values = [avg_metrics[m][metric] for m in methods]
        bars = ax.bar(method_names, values, color=colors, edgecolor='black', linewidth=1.2)
        ax.set_ylabel(metric, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis='both', labelsize=11)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                   f'{val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.suptitle('Retrieval Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/retrieval_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{OUTPUT_DIR}/retrieval_comparison.pdf", bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/retrieval_comparison.png")
    
    # Figure 2: Improvement from reranking
    fig, ax = plt.subplots(figsize=(10, 6))
    
    improvements = {
        "P@1": avg_metrics["hybrid_reranked"]["P@1"] - avg_metrics["rrf"]["P@1"],
        "P@3": avg_metrics["hybrid_reranked"]["P@3"] - avg_metrics["rrf"]["P@3"],
        "P@5": avg_metrics["hybrid_reranked"]["P@5"] - avg_metrics["rrf"]["P@5"],
        "MRR": avg_metrics["hybrid_reranked"]["MRR"] - avg_metrics["rrf"]["MRR"],
    }
    
    colors_imp = ['#27ae60' if v >= 0 else '#e74c3c' for v in improvements.values()]
    bars = ax.bar(improvements.keys(), improvements.values(), color=colors_imp, edgecolor='black', linewidth=1.2)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylabel('Improvement over RRF', fontsize=14, fontweight='bold')
    ax.set_title('Cross-Encoder Reranking Impact', fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12)
    
    for bar, val in zip(bars, improvements.values()):
        ypos = bar.get_height() + 0.01 if val >= 0 else bar.get_height() - 0.03
        ax.text(bar.get_x() + bar.get_width()/2, ypos, f'{val:+.2f}', 
               ha='center', va='bottom' if val >= 0 else 'top', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/reranking_improvement.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{OUTPUT_DIR}/reranking_improvement.pdf", bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/reranking_improvement.png")
    
    # Figure 3: Pipeline diagram data (for manual creation or description)
    pipeline_stats = {
        "Corpus": "1,131 transcripts",
        "Chunks": "28,631 chunks",
        "Embedding Dim": "1,024 (BGE-large)",
        "FAISS Index": "112 MB",
        "BM25 Index": "125 MB",
        "Reranker": "ms-marco-MiniLM"
    }
    
    print("\nPipeline Statistics (for poster):")
    for k, v in pipeline_stats.items():
        print(f"  {k}: {v}")
    
    # Figure 4: Per-query breakdown heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    
    query_labels = [q[0][:30] + "..." for q in test_queries]
    metric_data = np.array([[all_metrics[m][i]["MRR"] for m in methods] for i in range(len(test_queries))])
    
    sns.heatmap(metric_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                xticklabels=method_names, yticklabels=query_labels,
                ax=ax, vmin=0, vmax=1, linewidths=0.5)
    ax.set_title('MRR by Query and Method', fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', labelsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/mrr_heatmap.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{OUTPUT_DIR}/mrr_heatmap.pdf", bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/mrr_heatmap.png")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE - Figures saved to:", OUTPUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()
