# FinVERIFY: Multi-Aspect Retrieval-Augmented Financial Fact-Checking

[![NYU](https://img.shields.io/badge/NYU-NLP%20Final%20Project-purple)](https://www.nyu.edu/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📌 Overview

**FinVERIFY** is a retrieval-augmented generation (RAG) system for financial fact-checking, implementing the MAINRAG multi-aspect retrieval approach. The system retrieves evidence from earnings call transcripts and generates verified answers with citations.

### Key Features
- **4-Aspect Retrieval (MAINRAG)**: Semantic + Lexical + Entity + Temporal
- **Reciprocal Rank Fusion (RRF)**: Combines multiple retrieval signals
- **Cross-Encoder Reranking**: Precision refinement with MiniLM
- **Answer Generation**: Flan-T5 with retrieved context

## 🏗️ Architecture
```
Query → [MAINRAG Multi-Aspect Retrieval] → [RRF Fusion] → [Cross-Encoder Reranking] → [Flan-T5 Generation] → Answer + Citations
         ├── Semantic (FAISS + BGE)
         ├── Lexical (BM25)
         ├── Entity Filtering
         └── Temporal Filtering
```

## 📊 Dataset

| Metric | Value |
|--------|-------|
| Source | Kaggle Earnings Call Transcripts |
| Companies | 48 |
| Transcripts | 1,131 |
| Chunks | 28,631 |
| Year Range | 2007-2025 |
| Total Size | 193 MB |

## 🔬 Results

### Retrieval Performance

| Method | P@5 | MRR | Hit@1 |
|--------|-----|-----|-------|
| FAISS (Semantic) | 0.825 | 0.906 | - |
| BM25 (Lexical) | 0.875 | 0.893 | - |
| RRF (Hybrid) | 0.900 | 1.000 | - |
| **2-Aspect + Reranker** | 0.560 | 0.767 | 0.600 |
| **4-Aspect MAINRAG** | 0.580 | 0.717 | 0.600 |

### Key Findings
- RRF achieves **perfect MRR (1.0)** on basic retrieval
- MAINRAG improves **P@5 by 3.6%** over 2-aspect baseline
- Cross-encoder reranking finds exact matches (e.g., Apple Q4 2023)

### Sample QA Results

| Query | Answer | Top Citation |
|-------|--------|--------------|
| Apple's iPhone revenue Q4 2023? | $43.8 billion | Apple 2023 Q4 |
| NVIDIA data center growth? | 427% YoY | Nvidia 2025 Q1 |
| Microsoft Azure growth rate? | 30-31% | Microsoft 2020 Q4 |
| Amazon AWS operating margin? | 29.6% | Amazon 2023 Q4 |
| Walmart e-commerce performance? | 24% climb | Walmart 2024 Q3 |

## 📁 Project Structure
```
finverify_2/
├── data/
│   ├── raw/                    # Earnings call transcripts
│   ├── processed/
│   │   ├── chunks/             # 28,631 text chunks
│   │   ├── embeddings.npy      # BGE-large embeddings (1024-dim)
│   │   └── metadata.csv        # Transcript metadata
│   ├── indexes/
│   │   ├── faiss_index.bin     # FAISS vector index
│   │   └── bm25_index.pkl      # BM25 inverted index
│   └── outputs/                # Results and figures
├── src/
│   ├── ingestion/              # Data loading
│   ├── chunking/               # Text chunking (512 tokens, 50 overlap)
│   ├── embeddings/             # BGE embedding generation
│   ├── bm25/                   # BM25 index building
│   ├── rag/                    # MAINRAG hybrid retrieval
│   ├── reranker/               # Cross-encoder reranking
│   ├── generator/              # Flan-T5 answer generation
│   └── evaluation/             # Metrics and visualization
├── results/
│   ├── figures/                # Poster visualizations
│   ├── qa_results.json         # QA outputs
│   └── comparison_results.json # Evaluation metrics
└── README.md
```

## 🛠️ Technologies

| Component | Technology |
|-----------|------------|
| Embeddings | BGE-large-en-v1.5 (1024-dim) |
| Vector Search | FAISS (IndexFlatIP) |
| Lexical Search | BM25 (Rank-BM25) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Generator | google/flan-t5-base |
| Compute | NYU HPC (NVIDIA L4 GPU) |

## 🚀 Usage

### Installation
```bash
pip install torch sentence-transformers faiss-gpu rank-bm25 transformers
```

### Quick Start
```python
from src.generator.answer_generator import FinVerifyRAG

rag = FinVerifyRAG()
response = rag.answer("What was Apple's iPhone revenue in Q4 2023?")
print(response["answer"])  # $43.8 billion
print(response["citations"])  # [Apple 2023 Q4, ...]
```

## 📈 Poster Figures

| Figure | Description |
|--------|-------------|
| `retrieval_comparison.png` | FAISS vs BM25 vs RRF vs Reranked |
| `reranking_improvement.png` | Cross-encoder impact |
| `mrr_heatmap.png` | Per-query performance breakdown |
| `mainrag_comparison.png` | 2-aspect vs 4-aspect comparison |

## 👥 Team

| Member | Contribution |
|--------|--------------|
| Shashank Dugad | Data pipeline, embeddings, BM25, demo, poster |
| Utkarsh Arora | (Planned) Embeddings, BM25 |
| Shivam Balikondwar | (Planned) FAISS, reranking, evaluation |
| Surbhi | (Planned) MAINRAG, RRF, T5, report |

## 📚 References

1. Wang et al. (2025). "Multi-Aspect Integration for Enhanced RAG" (MAINRAG)
2. Lewis et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP"
3. Kaggle Earnings Call Transcripts Dataset

## 📄 License

MIT License - NYU NLP Final Project Fall 2025

---
*NYU Center for Data Science | DS-GA 1011 NLP*
