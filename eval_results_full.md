| system | Hit@1 | Hit@3 | MRR | abstain (neg) | p50 ms | p95 ms | flagship | generic | long | prefix | product | translit | typo | xlang |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BM25F only (no boosts, no prefix/typo/translit) | 82.2% | 87.7% | 0.857 | 80% | 0.08 | 0.12 | 71% | 67% | 100% | 100% | 100% | 20% | 40% | 95% |
| BM25F + prefix/typo/translit | 90.4% | 97.3% | 0.937 | 80% | 0.10 | 0.29 | 79% | 67% | 100% | 100% | 100% | 80% | 60% | 100% |
| Ours: + curated aliases, pins, boosts | 98.6% | 100.0% | 0.993 | 80% | 0.10 | 0.30 | 100% | 100% | 100% | 100% | 100% | 100% | 80% | 100% |
| Embeddings (multilingual MiniLM, reference only) | 56.2% | 74.0% | 0.648 | 100% | 29.09 | 47.76 | 29% | 100% | 50% | 17% | 65% | 40% | 40% | 84% |
