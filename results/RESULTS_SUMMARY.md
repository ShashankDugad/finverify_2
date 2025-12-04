# FinVERIFY Results Summary

## Evaluation Metrics

### 2-Aspect vs 4-Aspect (MAINRAG)
- **P@5**: 0.560 → 0.580 (+3.6%)
- **MRR**: 0.767 → 0.717 (-6.5%)
- **Hit@1**: 0.600 → 0.600 (equal)

### Retrieval Methods Comparison
| Method | P@5 | R@10 | MRR |
|--------|-----|------|-----|
| FAISS | 0.825 | 0.019 | 0.906 |
| BM25 | 0.875 | 0.016 | 0.893 |
| RRF | 0.900 | 0.016 | 1.000 |
| RRF + Reranker | 0.850 | 0.015 | 1.000 |

## Pipeline Statistics

| Component | Value |
|-----------|-------|
| Total transcripts | 1,131 |
| Total chunks | 28,631 |
| Embedding dimension | 1,024 |
| Chunk size | 512 tokens |
| Overlap | 50 tokens |
| Companies | 48 |
| Year range | 2007-2025 |

## QA Sample Outputs

1. **Apple iPhone Q4 2023**: $43.8 billion
2. **NVIDIA data center growth**: 427% YoY
3. **Microsoft Azure growth**: 30-31%
4. **Amazon AWS margin**: 29.6%
5. **Walmart e-commerce**: 24% climb

## Key Insights

1. **RRF fusion** achieves perfect MRR=1.0 on retrieval
2. **Cross-encoder reranking** finds exact quarter matches
3. **Entity/temporal filtering** helps specific queries
4. **Trade-off**: Stricter filtering can reduce recall
