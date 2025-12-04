import json
import numpy as np
import pickle
import faiss
import re
from sentence_transformers import SentenceTransformer, CrossEncoder
from collections import defaultdict

# Paths
FAISS_INDEX = "/scratch/sd5957/finverify_2/data/indexes/faiss_index.bin"
BM25_INDEX = "/scratch/sd5957/finverify_2/data/indexes/bm25_index.pkl"
CHUNKS_FILE = "/scratch/sd5957/finverify_2/data/processed/chunks/chunks.jsonl"
METADATA_FILE = "/scratch/sd5957/finverify_2/data/processed/chunk_metadata.jsonl"

# Company aliases - use word boundaries to avoid false matches
COMPANY_ALIASES = {
    # Tech
    r'\bapple\b': "Apple", r'\baapl\b': "Apple", r'\biphone\b': "Apple", r'\bipad\b': "Apple",
    r'\bnvidia\b': "Nvidia", r'\bnvda\b': "Nvidia", r'\bgeforce\b': "Nvidia",
    r'\bmicrosoft\b': "Microsoft", r'\bmsft\b': "Microsoft", r'\bazure\b': "Microsoft",
    r'\bamazon\b': "Amazon", r'\bamzn\b': "Amazon", r'\baws\b': "Amazon",
    r'\bgoogle\b': "Alphabet", r'\balphabet\b': "Alphabet", r'\bgoog\b': "Alphabet", r'\byoutube\b': "Alphabet",
    r'\bmeta\b': "META", r'\bfacebook\b': "META", r'\binstagram\b': "META",
    r'\bamd\b': "AMD", r'\bryzen\b': "AMD", r'\bradeon\b': "AMD",
    # Finance
    r'\bjpmorgan\b': "JPM", r'\bjpm\b': "JPM", r'\bchase\b': "JPM",
    r'\bciti\b': "Citi", r'\bcitibank\b': "Citi", r'\bcitigroup\b': "Citi",
    r'\bmastercard\b': "Mastercard",
    r'\bpaypal\b': "PAYPAL",
    # Retail
    r'\bwalmart\b': "Walmart", r'\bwmt\b': "Walmart",
    r'\bcostco\b': "Costco",
    r'\blululemon\b': "Lululemon",
    # Auto
    r'\bford\b': "Ford",
    r'\bgeneral motors\b': "GM", r'\bgm\b': "GM", r'\bchevrolet\b': "GM",
    r'\bbmw\b': "BMW",
    # Healthcare
    r'\bunitedhealth\b': "UnitedHealth",
    # Others
    r'\bnike\b': "Nike",
    r'\boracle\b': "Oracle",
    r'\bsalesforce\b': "SalesForce",
    r'\bibm\b': "IBM",
    r'\bcisco\b': "Cisco",
    r'\badobe\b': "Adobe",
    r'\baccenture\b': "Accenture",
    r'\bdisney\b': "Walt Disney",
    r'\bmarriott\b': "Marriott",
}

# Quarter patterns
QUARTER_PATTERNS = [
    (r'\bq1\b', 1), (r'\bq2\b', 2), (r'\bq3\b', 3), (r'\bq4\b', 4),
    (r'\bfirst quarter\b', 1), (r'\bsecond quarter\b', 2), 
    (r'\bthird quarter\b', 3), (r'\bfourth quarter\b', 4),
]

YEAR_PATTERN = re.compile(r'\b(20\d{2})\b')


class MAINRAGRetriever:
    """
    MAINRAG v2: Improved entity extraction with word boundaries
    """
    
    def __init__(self):
        print("=" * 60)
        print("Initializing MAINRAG v2 Multi-Aspect Retriever")
        print("=" * 60)
        
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
        
        print("Building entity index...")
        self.entity_index = defaultdict(set)
        for i, meta in enumerate(self.metadata):
            self.entity_index[meta["company"]].add(i)
        
        print("Building temporal index...")
        self.year_index = defaultdict(set)
        self.quarter_index = defaultdict(set)
        for i, meta in enumerate(self.metadata):
            year = str(meta["year"])
            quarter = str(meta["quarter"])
            self.year_index[year].add(i)
            self.quarter_index[(year, quarter)].add(i)
        
        print("Loading models...")
        self.bi_encoder = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda")
        
        print("=" * 60)
        print("MAINRAG v2 Ready!")
        print("=" * 60)
    
    def tokenize(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return [t for t in text.split() if len(t) > 1]
    
    def extract_entities(self, query):
        """Extract entities using word boundary regex."""
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
    
    def search_semantic(self, query, top_k=100):
        query_emb = self.bi_encoder.encode([query], normalize_embeddings=True).astype('float32')
        scores, indices = self.faiss_index.search(query_emb, top_k)
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]
    
    def search_lexical(self, query, top_k=100):
        tokens = self.tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = scores.argsort()[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]
    
    def search_entity(self, entities, top_k=200):
        if not entities:
            return []
        matching_docs = set()
        for entity in entities:
            if entity in self.entity_index:
                matching_docs.update(self.entity_index[entity])
        return [(doc_id, 1.0) for doc_id in list(matching_docs)[:top_k]]
    
    def search_temporal(self, year, quarter, top_k=200):
        if not year:
            return []
        if quarter:
            matching_docs = self.quarter_index.get((year, quarter), set())
        else:
            matching_docs = self.year_index.get(year, set())
        return [(doc_id, 1.0) for doc_id in list(matching_docs)[:top_k]]
    
    def reciprocal_rank_fusion(self, result_lists, k=60, weights=None):
        if weights is None:
            weights = [1.0] * len(result_lists)
        
        rrf_scores = defaultdict(float)
        for weight, results in zip(weights, result_lists):
            for rank, (doc_id, _) in enumerate(results):
                rrf_scores[doc_id] += weight * (1.0 / (k + rank + 1))
        
        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    def search(self, query, top_k=10, use_reranker=True, verbose=False):
        entities = self.extract_entities(query)
        year, quarter = self.extract_temporal(query)
        
        if verbose:
            print(f"\nQuery Analysis:")
            print(f"  Entities: {entities}")
            print(f"  Year: {year}, Quarter: {quarter}")
        
        semantic_results = self.search_semantic(query, top_k=100)
        lexical_results = self.search_lexical(query, top_k=100)
        entity_results = self.search_entity(entities, top_k=200)
        temporal_results = self.search_temporal(year, quarter, top_k=200)
        
        if verbose:
            print(f"\nAspect Results:")
            print(f"  Semantic: {len(semantic_results)} docs")
            print(f"  Lexical: {len(lexical_results)} docs")
            print(f"  Entity: {len(entity_results)} docs")
            print(f"  Temporal: {len(temporal_results)} docs")
        
        # Higher weights for entity and temporal when available
        active_results = [semantic_results, lexical_results]
        weights = [1.0, 1.0]
        
        if entity_results:
            active_results.append(entity_results)
            weights.append(2.0)  # Strong boost for entity
        
        if temporal_results:
            active_results.append(temporal_results)
            weights.append(2.0)  # Strong boost for temporal
        
        rrf_results = self.reciprocal_rank_fusion(active_results, weights=weights)
        
        if use_reranker:
            candidate_ids = [doc_id for doc_id, _ in rrf_results[:50]]
            pairs = [(query, self.chunks[doc_id]["text"][:512]) for doc_id in candidate_ids]
            ce_scores = self.cross_encoder.predict(pairs)
            reranked = sorted(zip(candidate_ids, ce_scores), key=lambda x: x[1], reverse=True)[:top_k]
        else:
            reranked = rrf_results[:top_k]
        
        results = []
        for doc_id, score in reranked:
            meta = self.metadata[doc_id]
            results.append({
                "doc_id": doc_id,
                "score": float(score),
                "company": meta["company"],
                "year": meta["year"],
                "quarter": meta["quarter"],
                "text": self.chunks[doc_id]["text"][:300] + "..."
            })
        
        return {
            "query": query,
            "extracted_entities": entities,
            "extracted_year": year,
            "extracted_quarter": quarter,
            "num_aspects_used": len(active_results),
            "results": results
        }


def test_mainrag():
    retriever = MAINRAGRetriever()
    
    test_queries = [
        "What was Apple's iPhone revenue in Q4 2023?",
        "How did NVIDIA's data center perform in Q1 2025?",
        "Microsoft Azure growth in 2024",
        "Amazon AWS profit margins",
        "Walmart e-commerce strategy",
        "Tech earnings Q3 2024",
        "Supply chain challenges in retail",
    ]
    
    print("\n" + "=" * 70)
    print("MAINRAG v2 MULTI-ASPECT RETRIEVAL TEST")
    print("=" * 70)
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"QUERY: {query}")
        print("=" * 70)
        
        response = retriever.search(query, top_k=5, verbose=True)
        
        print(f"\nAspects used: {response['num_aspects_used']}")
        print(f"Entities: {response['extracted_entities']}")
        print(f"Year: {response['extracted_year']}, Quarter: {response['extracted_quarter']}")
        
        print(f"\nTop 5 Results:")
        for i, r in enumerate(response["results"]):
            print(f"  {i+1}. [{r['company']} {r['year']} Q{r['quarter']}] score={r['score']:.4f}")
    
    print("\n" + "=" * 70)
    print("MAINRAG v2 TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_mainrag()
