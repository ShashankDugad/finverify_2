import json
import numpy as np
import pickle
import faiss
import re
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
plt.style.use('seaborn-v0_8-whitegrid')

# Entity aliases with word boundaries
COMPANY_ALIASES = {
    r'\bapple\b': "Apple", r'\biphone\b': "Apple",
    r'\bnvidia\b': "Nvidia", r'\bgeforce\b': "Nvidia",
    r'\bmicrosoft\b': "Microsoft", r'\bazure\b': "Microsoft",
    r'\bamazon\b': "Amazon", r'\baws\b': "Amazon",
    r'\bgoogle\b': "Alphabet", r'\balphabet\b': "Alphabet",
    r'\bmeta\b': "META", r'\bfacebook\b': "META",
    r'\bwalmart\b': "Walmart",
    r'\bcostco\b': "Costco",
    r'\bnike\b': "Nike",
    r'\boracle\b': "Oracle",
    r'\bibm\b': "IBM",
    r'\badobe\b': "Adobe",
}

QUARTER_PATTERNS = [
    (r'\bq1\b', 1), (r'\bq2\b', 2), (r'\bq3\b', 3), (r'\bq4\b', 4),
]
YEAR_PATTERN = re.compile(r'\b(20\d{2})\b')


class MethodComparison:
    def __init__(self):
        print("Loading components...")
        self.faiss_index = faiss.read_index(FAISS_INDEX)
        
        with open(BM25_INDEX, "rb") as f:
            self.bm25 = pickle.load(f)["bm25"]
        
        self.chunks = []
        with open(CHUNKS_FILE) as f:
            for line in f:
                self.chunks.append(json.loads(line))
        
        self.metadata = []
        with open(METADATA_FILE) as f:
            for line in f:
                self.metadata.append(json.loads(line))
        
        # Build indexes
        self.entity_index = defaultdict(set)
        self.year_index = defaultdict(set)
        self.quarter_index = defaultdict(set)
        for i, m in enumerate(self.metadata):
            self.entity_index[m["company"]].add(i)
            self.year_index[str(m["year"])].add(i)
            self.quarter_index[(str(m["year"]), str(m["quarter"]))].add(i)
        
        print("Loading models...")
        self.bi_encoder = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda")
        print("Ready!")
    
    def tokenize(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return [t for t in text.split() if len(t) > 1]
    
    def extract_entities(self, query):
        query_lower = query.lower()
        entities = set()
        for pattern, company in COMPANY_ALIASES.items():
            if re.search(pattern, query_lower):
                entities.add(company)
        return list(entities)
    
    def extract_temporal(self, query):
        query_lower = query.lower()
        year_match = YEAR_PATTERN.search(query)
        year = year_match.group(1) if year_match else None
        quarter = None
        for pattern, q in QUARTER_PATTERNS:
            if re.search(pattern, query_lower):
                quarter = str(q)
                break
        return year, quarter
    
    def rrf(self, result_lists, k=60, weights=None):
        if weights is None:
            weights = [1.0] * len(result_lists)
        scores = defaultdict(float)
        for w, results in zip(weights, result_lists):
            for rank, (doc_id, _) in enumerate(results):
                scores[doc_id] += w / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    def search_2aspect(self, query, top_k=10):
        """2-aspect: FAISS + BM25 only"""
        # FAISS
        q_emb = self.bi_encoder.encode([query], normalize_embeddings=True).astype('float32')
        _, faiss_ids = self.faiss_index.search(q_emb, 100)
        faiss_results = [(int(i), 1.0) for i in faiss_ids[0]]
        
        # BM25
        bm25_scores = self.bm25.get_scores(self.tokenize(query))
        bm25_ids = bm25_scores.argsort()[::-1][:100]
        bm25_results = [(int(i), 1.0) for i in bm25_ids]
        
        # RRF
        rrf_results = self.rrf([faiss_results, bm25_results])
        
        # Rerank
        candidates = [d for d, _ in rrf_results[:50]]
        pairs = [(query, self.chunks[d]["text"][:512]) for d in candidates]
        ce_scores = self.cross_encoder.predict(pairs)
        reranked = sorted(zip(candidates, ce_scores), key=lambda x: x[1], reverse=True)
        
        return [d for d, _ in reranked[:top_k]]
    
    def search_4aspect(self, query, top_k=10):
        """4-aspect: FAISS + BM25 + Entity + Temporal"""
        entities = self.extract_entities(query)
        year, quarter = self.extract_temporal(query)
        
        # FAISS
        q_emb = self.bi_encoder.encode([query], normalize_embeddings=True).astype('float32')
        _, faiss_ids = self.faiss_index.search(q_emb, 100)
        faiss_results = [(int(i), 1.0) for i in faiss_ids[0]]
        
        # BM25
        bm25_scores = self.bm25.get_scores(self.tokenize(query))
        bm25_ids = bm25_scores.argsort()[::-1][:100]
        bm25_results = [(int(i), 1.0) for i in bm25_ids]
        
        # Entity
        entity_docs = set()
        for e in entities:
            entity_docs.update(self.entity_index.get(e, set()))
        entity_results = [(d, 1.0) for d in list(entity_docs)[:200]]
        
        # Temporal
        if year and quarter:
            temporal_docs = self.quarter_index.get((year, quarter), set())
        elif year:
            temporal_docs = self.year_index.get(year, set())
        else:
            temporal_docs = set()
        temporal_results = [(d, 1.0) for d in list(temporal_docs)[:200]]
        
        # RRF with weights
        results = [faiss_results, bm25_results]
        weights = [1.0, 1.0]
        if entity_results:
            results.append(entity_results)
            weights.append(2.0)
        if temporal_results:
            results.append(temporal_results)
            weights.append(2.0)
        
        rrf_results = self.rrf(results, weights=weights)
        
        # Rerank
        candidates = [d for d, _ in rrf_results[:50]]
        pairs = [(query, self.chunks[d]["text"][:512]) for d in candidates]
        ce_scores = self.cross_encoder.predict(pairs)
        reranked = sorted(zip(candidates, ce_scores), key=lambda x: x[1], reverse=True)
        
        return [d for d, _ in reranked[:top_k]]
    
    def is_relevant(self, doc_id, company, year=None, quarter=None):
        m = self.metadata[doc_id]
        if m["company"].lower() != company.lower():
            return False
        if year and str(m["year"]) != str(year):
            return False
        if quarter and str(m["quarter"]) != str(quarter):
            return False
        return True
    
    def compute_metrics(self, retrieved, company, year=None, quarter=None):
        """Compute P@5, MRR, Hit@1"""
        # P@5
        relevant = sum(1 for d in retrieved[:5] if self.is_relevant(d, company, year, quarter))
        p5 = relevant / 5
        
        # MRR
        mrr = 0
        for rank, d in enumerate(retrieved):
            if self.is_relevant(d, company, year, quarter):
                mrr = 1.0 / (rank + 1)
                break
        
        # Hit@1
        hit1 = 1 if retrieved and self.is_relevant(retrieved[0], company, year, quarter) else 0
        
        return {"P@5": p5, "MRR": mrr, "Hit@1": hit1}


def run_comparison():
    comp = MethodComparison()
    
    # Test queries with ground truth
    test_queries = [
        ("What was Apple's iPhone revenue in Q4 2023?", "Apple", "2023", "4"),
        ("NVIDIA data center growth Q1 2025", "Nvidia", "2025", "1"),
        ("Microsoft Azure revenue 2024", "Microsoft", "2024", None),
        ("Amazon AWS profit margins", "Amazon", None, None),
        ("Walmart e-commerce sales growth", "Walmart", None, None),
        ("Meta advertising revenue Q3 2023", "META", "2023", "3"),
        ("Oracle cloud revenue 2024", "Oracle", "2024", None),
        ("Nike quarterly earnings Q2 2024", "Nike", "2024", "2"),
        ("IBM AI revenue growth", "IBM", None, None),
        ("Adobe Creative Cloud 2024", "Adobe", "2024", None),
    ]
    
    results_2aspect = []
    results_4aspect = []
    
    print("\n" + "=" * 70)
    print("2-ASPECT vs 4-ASPECT (MAINRAG) COMPARISON")
    print("=" * 70)
    
    for query, company, year, quarter in test_queries:
        print(f"\nQuery: {query[:50]}...")
        
        # 2-aspect
        r2 = comp.search_2aspect(query, top_k=10)
        m2 = comp.compute_metrics(r2, company, year, quarter)
        results_2aspect.append(m2)
        
        # 4-aspect
        r4 = comp.search_4aspect(query, top_k=10)
        m4 = comp.compute_metrics(r4, company, year, quarter)
        results_4aspect.append(m4)
        
        print(f"  2-Aspect: P@5={m2['P@5']:.2f} MRR={m2['MRR']:.2f} Hit@1={m2['Hit@1']}")
        print(f"  4-Aspect: P@5={m4['P@5']:.2f} MRR={m4['MRR']:.2f} Hit@1={m4['Hit@1']}")
    
    # Average metrics
    avg_2 = {k: np.mean([r[k] for r in results_2aspect]) for k in ["P@5", "MRR", "Hit@1"]}
    avg_4 = {k: np.mean([r[k] for r in results_4aspect]) for k in ["P@5", "MRR", "Hit@1"]}
    
    print("\n" + "=" * 70)
    print("AVERAGE RESULTS")
    print("=" * 70)
    print(f"2-Aspect (FAISS+BM25):    P@5={avg_2['P@5']:.3f}  MRR={avg_2['MRR']:.3f}  Hit@1={avg_2['Hit@1']:.3f}")
    print(f"4-Aspect (MAINRAG):       P@5={avg_4['P@5']:.3f}  MRR={avg_4['MRR']:.3f}  Hit@1={avg_4['Hit@1']:.3f}")
    
    improvement = {k: (avg_4[k] - avg_2[k]) / avg_2[k] * 100 if avg_2[k] > 0 else 0 for k in avg_2}
    print(f"\nImprovement:              P@5={improvement['P@5']:+.1f}%  MRR={improvement['MRR']:+.1f}%  Hit@1={improvement['Hit@1']:+.1f}%")
    
    # ==================== VISUALIZATION ====================
    print("\n" + "=" * 70)
    print("GENERATING COMPARISON FIGURE")
    print("=" * 70)
    
    # Figure: Bar chart comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ["P@5", "MRR", "Hit@1"]
    x = np.arange(len(metrics))
    width = 0.35
    
    vals_2 = [avg_2[m] for m in metrics]
    vals_4 = [avg_4[m] for m in metrics]
    
    bars1 = ax.bar(x - width/2, vals_2, width, label='2-Aspect\n(FAISS+BM25)', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, vals_4, width, label='4-Aspect\n(MAINRAG)', color='#e74c3c', edgecolor='black')
    
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('2-Aspect vs 4-Aspect (MAINRAG) Retrieval', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11, loc='upper right')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f'{h:.2f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/mainrag_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{OUTPUT_DIR}/mainrag_comparison.pdf", bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/mainrag_comparison.png")
    
    # Save results JSON
    results_json = {
        "2_aspect": avg_2,
        "4_aspect": avg_4,
        "improvement_pct": improvement,
        "queries": len(test_queries)
    }
    with open(f"{OUTPUT_DIR}/comparison_results.json", "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Saved: {OUTPUT_DIR}/comparison_results.json")
    
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_comparison()
