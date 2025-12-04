import json
import numpy as np
import pickle
import faiss
import re
import torch
import gradio as gr
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import T5Tokenizer, T5ForConditionalGeneration
from collections import defaultdict

# Paths
FAISS_INDEX = "/scratch/sd5957/finverify_2/data/indexes/faiss_index.bin"
BM25_INDEX = "/scratch/sd5957/finverify_2/data/indexes/bm25_index.pkl"
CHUNKS_FILE = "/scratch/sd5957/finverify_2/data/processed/chunks/chunks.jsonl"
METADATA_FILE = "/scratch/sd5957/finverify_2/data/processed/chunk_metadata.jsonl"

# Entity aliases
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

QUARTER_PATTERNS = [(r'\bq1\b', 1), (r'\bq2\b', 2), (r'\bq3\b', 3), (r'\bq4\b', 4)]
YEAR_PATTERN = re.compile(r'\b(20\d{2})\b')


class FinVerifyDemo:
    def __init__(self):
        print("Loading FinVERIFY components...")
        self.faiss_index = faiss.read_index(FAISS_INDEX)
        with open(BM25_INDEX, "rb") as f:
            self.bm25 = pickle.load(f)["bm25"]
        
        self.chunks = [json.loads(line) for line in open(CHUNKS_FILE)]
        self.metadata = [json.loads(line) for line in open(METADATA_FILE)]
        
        self.entity_index = defaultdict(set)
        self.year_index = defaultdict(set)
        self.quarter_index = defaultdict(set)
        for i, m in enumerate(self.metadata):
            self.entity_index[m["company"]].add(i)
            self.year_index[str(m["year"])].add(i)
            self.quarter_index[(str(m["year"]), str(m["quarter"]))].add(i)
        
        self.bi_encoder = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda")
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        self.generator = T5ForConditionalGeneration.from_pretrained(
            "google/flan-t5-base", device_map="cuda", torch_dtype=torch.float16)
        print("FinVERIFY ready!")
    
    def tokenize(self, text):
        return [t for t in re.sub(r'[^a-z0-9\s]', ' ', text.lower()).split() if len(t) > 1]
    
    def extract_entities(self, query):
        return list({c for p, c in COMPANY_ALIASES.items() if re.search(p, query.lower())})
    
    def extract_temporal(self, query):
        year = (m.group(1) if (m := YEAR_PATTERN.search(query)) else None)
        quarter = next((str(q) for p, q in QUARTER_PATTERNS if re.search(p, query.lower())), None)
        return year, quarter
    
    def rrf(self, result_lists, weights=None, k=60):
        weights = weights or [1.0] * len(result_lists)
        scores = defaultdict(float)
        for w, results in zip(weights, result_lists):
            for rank, (doc_id, _) in enumerate(results):
                scores[doc_id] += w / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    def retrieve(self, query, use_mainrag=True, top_k=5):
        entities = self.extract_entities(query) if use_mainrag else []
        year, quarter = self.extract_temporal(query) if use_mainrag else (None, None)
        
        q_emb = self.bi_encoder.encode([query], normalize_embeddings=True).astype('float32')
        _, faiss_ids = self.faiss_index.search(q_emb, 100)
        faiss_results = [(int(i), 1.0) for i in faiss_ids[0]]
        
        bm25_scores = self.bm25.get_scores(self.tokenize(query))
        bm25_results = [(int(i), 1.0) for i in bm25_scores.argsort()[::-1][:100]]
        
        results, weights = [faiss_results, bm25_results], [1.0, 1.0]
        
        if use_mainrag and entities:
            entity_docs = set().union(*[self.entity_index.get(e, set()) for e in entities])
            if entity_docs:
                results.append([(d, 1.0) for d in list(entity_docs)[:200]])
                weights.append(2.0)
        
        if use_mainrag and year:
            temporal_docs = self.quarter_index.get((year, quarter), set()) if quarter else self.year_index.get(year, set())
            if temporal_docs:
                results.append([(d, 1.0) for d in list(temporal_docs)[:200]])
                weights.append(2.0)
        
        rrf_results = self.rrf(results, weights)
        candidates = [d for d, _ in rrf_results[:50]]
        ce_scores = self.cross_encoder.predict([(query, self.chunks[d]["text"][:512]) for d in candidates])
        return sorted(zip(candidates, ce_scores), key=lambda x: x[1], reverse=True)[:top_k], entities, year, quarter
    
    def generate(self, query, contexts):
        prompt = f"Context:\n{chr(10).join(contexts[:3])}\n\nQuestion: {query}\n\nAnswer:"
        inputs = {k: v.to("cuda") for k, v in self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).items()}
        with torch.no_grad():
            outputs = self.generator.generate(**inputs, max_length=256, num_beams=4)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def query(self, question, use_mainrag=True):
        if not question.strip():
            return "Please enter a question.", "", ""
        results, entities, year, quarter = self.retrieve(question, use_mainrag)
        
        info = f"**Entities:** {entities or 'None'} | **Year:** {year or 'None'} | **Quarter:** Q{quarter or 'None'}\n\n"
        contexts = []
        for i, (doc_id, score) in enumerate(results):
            m = self.metadata[doc_id]
            contexts.append(f"[{m['company']} {m['year']} Q{m['quarter']}]: {self.chunks[doc_id]['text'][:400]}")
            info += f"{i+1}. **{m['company']} {m['year']} Q{m['quarter']}** (score: {score:.2f})\n"
        
        answer = self.generate(question, contexts)
        citations = "\n".join([f"[{i+1}] {self.metadata[d]['company']} {self.metadata[d]['year']} Q{self.metadata[d]['quarter']}" for i, (d, _) in enumerate(results)])
        return answer, info, citations


demo_instance = FinVerifyDemo()

def process_query(question, use_mainrag):
    return demo_instance.query(question, use_mainrag)

with gr.Blocks(title="FinVERIFY") as demo:
    gr.Markdown("# 🔍 FinVERIFY: Financial Fact-Checking\nAsk questions about company earnings from **1,131 transcripts**.")
    
    with gr.Row():
        question_input = gr.Textbox(label="Question", placeholder="What was Apple's iPhone revenue in Q4 2023?", lines=2)
        use_mainrag = gr.Checkbox(label="Use MAINRAG (4-aspect)", value=True)
    
    submit_btn = gr.Button("Get Answer", variant="primary")
    answer_output = gr.Textbox(label="Answer", lines=2)
    citations_output = gr.Textbox(label="Citations", lines=3)
    
    with gr.Accordion("Retrieval Details", open=False):
        retrieval_output = gr.Markdown()
    
    submit_btn.click(process_query, [question_input, use_mainrag], [answer_output, retrieval_output, citations_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
