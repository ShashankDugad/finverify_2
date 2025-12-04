# FinVERIFY: Multi-Aspect Retrieval-Augmented Financial Fact-Checking

[![NYU](https://img.shields.io/badge/NYU-CDS-purple)](https://cds.nyu.edu/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)

A RAG system for financial fact-checking combining **semantic, lexical, entity, and temporal** retrieval signals.

## 🎯 Research Question

> How can we improve retrieval for financial QA by combining semantic, lexical, entity, and temporal signals?

## 📊 Results (100-Query Test Bench)

| Metric | Score |
|--------|-------|
| **Citation Accuracy** | **91%** |
| **F1 Score** | 0.33 |
| **Exact Match (EM)** | 9% |

### Retrieval Performance

| Method | P@5 | MRR |
|--------|-----|-----|
| FAISS (Semantic) | 0.82 | 0.91 |
| BM25 (Lexical) | 0.88 | 0.89 |
| **RRF (Hybrid)** | **0.90** | **1.00** |

### MAINRAG Comparison

| Method | P@5 | Improvement |
|--------|-----|-------------|
| 2-Aspect (FAISS+BM25) | 0.56 | - |
| **4-Aspect (MAINRAG)** | **0.58** | **+3.6%** |

### Sample QA Outputs

| Query | Prediction | Citation | F1 |
|-------|------------|----------|-----|
| Apple iPhone revenue Q4 2023? | $43.8 billion | Apple 2023 Q4 | 1.0 ✓ |
| Apple services revenue Q4 2023? | $22.3 billion | Apple 2023 Q4 | 1.0 ✓ |
| Oracle cloud revenue? | $5.6 billion | Oracle 2024 Q4 | 1.0 ✓ |
| Walmart e-commerce sales? | 24% climb | Walmart 2024 Q3 | 0.5 |
| Salesforce revenue? | $9.13 billion | SalesForce 2025 Q1 | 0.5 |

## 📁 Dataset

| Stat | Value |
|------|-------|
| **Companies** | 48 (Apple, NVIDIA, Microsoft, etc.) |
| **Transcripts** | 1,131 |
| **Chunks** | 28,631 (512 tokens) |
| **Years** | 2007-2025 |

## 🛠️ Tech Stack

- **Embeddings**: BGE-large-en-v1.5 (1024-dim)
- **Dense Index**: FAISS
- **Sparse Index**: BM25
- **Reranker**: MiniLM Cross-Encoder
- **Generator**: Flan-T5-base
- **Compute**: NYU HPC (NVIDIA L4)

## 🚀 Quick Start
```bash
# Run CLI Demo
python3 demo/cli_demo.py

# Run Test Bench
python3 src/evaluation/test_bench_100.py
```

## 📈 Key Findings

1. **RRF achieves MRR = 1.0** — Hybrid retrieval finds correct documents
2. **MAINRAG improves P@5 by 3.6%** — Entity/temporal filtering helps
3. **91% citation accuracy** — Retrieved docs match expected companies

## 👥 Team

- **Shashank Dugad** - Data pipeline, demo, poster
- **Utkarsh Arora** - Embeddings, BM25
- **Shivam Balikondwar** - FAISS, reranking, evaluation
- **Surbhi** - MAINRAG, RRF, T5

## 📚 References

- Wang et al. (2025). "Multi-Aspect Integration for Enhanced RAG"
- Lewis et al. (2020). "Retrieval-Augmented Generation"

---
**NYU CDS NLP Fall 2025**
