import json
import numpy as np
import pickle
import faiss
import re
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import T5Tokenizer, T5ForConditionalGeneration
from collections import defaultdict

# Paths
FAISS_INDEX = "/scratch/sd5957/finverify_2/data/indexes/faiss_index.bin"
BM25_INDEX = "/scratch/sd5957/finverify_2/data/indexes/bm25_index.pkl"
CHUNKS_FILE = "/scratch/sd5957/finverify_2/data/processed/chunks/chunks.jsonl"
METADATA_FILE = "/scratch/sd5957/finverify_2/data/processed/chunk_metadata.jsonl"


class FinVerifyRAG:
    """
    Complete RAG pipeline for financial fact-checking:
    Query → FAISS + BM25 → RRF → Cross-Encoder → T5 Generation → Answer
    """
    
    def __init__(self):
        print("=" * 60)
        print("Initializing FinVERIFY RAG Pipeline")
        print("=" * 60)
        
        # Load indexes
        print("Loading FAISS index...")
        self.faiss_index = faiss.read_index(FAISS_INDEX)
        
        print("Loading BM25 index...")
        with open(BM25_INDEX, "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25 = bm25_data["bm25"]
        
        # Load data
        print("Loading chunks and metadata...")
        self.chunks = []
        with open(CHUNKS_FILE, "r") as f:
            for line in f:
                self.chunks.append(json.loads(line))
        
        self.metadata = []
        with open(METADATA_FILE, "r") as f:
            for line in f:
                self.metadata.append(json.loads(line))
        
        # Load models
        print("Loading BGE bi-encoder...")
        self.bi_encoder = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")
        
        print("Loading Cross-Encoder reranker...")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda")
        
        print("Loading Flan-T5-base generator...")
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        self.generator = T5ForConditionalGeneration.from_pretrained(
            "google/flan-t5-base",
            device_map="cuda",
            torch_dtype=torch.float16
        )
        
        print("=" * 60)
        print("FinVERIFY RAG Pipeline Ready!")
        print("=" * 60)
    
    def tokenize(self, text):
        """BM25 tokenization."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return [t for t in text.split() if len(t) > 1]
    
    def retrieve(self, query, top_k=5):
        """
        Multi-aspect retrieval: FAISS + BM25 → RRF → Cross-Encoder
        """
        # Stage 1: FAISS (semantic)
        query_emb = self.bi_encoder.encode([query], normalize_embeddings=True).astype('float32')
        _, faiss_ids = self.faiss_index.search(query_emb, 100)
        
        # Stage 2: BM25 (lexical)
        bm25_scores = self.bm25.get_scores(self.tokenize(query))
        bm25_ids = bm25_scores.argsort()[::-1][:100]
        
        # Stage 3: RRF fusion
        rrf_scores = defaultdict(float)
        for rank, idx in enumerate(faiss_ids[0]):
            rrf_scores[idx] += 1.0 / (60 + rank + 1)
        for rank, idx in enumerate(bm25_ids):
            rrf_scores[idx] += 1.0 / (60 + rank + 1)
        
        rrf_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:50]
        
        # Stage 4: Cross-encoder reranking
        candidate_ids = [doc_id for doc_id, _ in rrf_results]
        pairs = [(query, self.chunks[doc_id]["text"][:512]) for doc_id in candidate_ids]
        ce_scores = self.cross_encoder.predict(pairs)
        
        reranked = sorted(zip(candidate_ids, ce_scores), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Build results
        results = []
        for doc_id, score in reranked:
            results.append({
                "doc_id": doc_id,
                "score": float(score),
                "metadata": self.metadata[doc_id],
                "text": self.chunks[doc_id]["text"]
            })
        
        return results
    
    def generate_answer(self, query, context, max_length=256):
        """
        Generate answer using Flan-T5 with retrieved context.
        """
        # Build prompt
        prompt = f"""Based on the following financial earnings call transcript, answer the question.

Context:
{context}

Question: {query}

Answer:"""
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.generator.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    
    def answer(self, query, top_k=3):
        """
        Full RAG pipeline: Retrieve → Generate → Answer with citations
        """
        # Retrieve relevant documents
        retrieved = self.retrieve(query, top_k=top_k)
        
        # Build context from top documents
        context_parts = []
        for i, doc in enumerate(retrieved):
            meta = doc["metadata"]
            source = f"[{meta['company']} {meta['year']} Q{meta['quarter']}]"
            context_parts.append(f"{source}: {doc['text'][:500]}")
        
        context = "\n\n".join(context_parts)
        
        # Generate answer
        answer = self.generate_answer(query, context)
        
        # Build response with citations
        citations = []
        for doc in retrieved:
            meta = doc["metadata"]
            citations.append({
                "source": f"{meta['company']} {meta['year']} Q{meta['quarter']}",
                "score": doc["score"]
            })
        
        return {
            "query": query,
            "answer": answer,
            "citations": citations,
            "num_sources": len(retrieved)
        }


def test_rag():
    """Test the complete RAG pipeline."""
    rag = FinVerifyRAG()
    
    test_queries = [
        "What was Apple's iPhone revenue in Q4 2023?",
        "How much did NVIDIA's data center revenue grow?",
        "What is Microsoft Azure's revenue growth rate?",
        "What was Amazon's AWS operating margin?",
        "How did Walmart's e-commerce sales perform?"
    ]
    
    print("\n" + "=" * 70)
    print("FinVERIFY RAG - QUESTION ANSWERING TEST")
    print("=" * 70)
    
    results = []
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"QUERY: {query}")
        print("=" * 70)
        
        response = rag.answer(query, top_k=3)
        
        print(f"\nANSWER: {response['answer']}")
        print(f"\nCITATIONS:")
        for i, cite in enumerate(response['citations']):
            print(f"   [{i+1}] {cite['source']} (score: {cite['score']:.2f})")
        
        results.append(response)
    
    # Save results
    output_file = "/scratch/sd5957/finverify_2/data/outputs/qa_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    print("\n" + "=" * 70)
    print("RAG PIPELINE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_rag()
