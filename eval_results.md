| system | Hit@1 | Hit@3 | MRR | abstain (neg) | p50 ms | p95 ms | flagship | generic | long | prefix | product | translit | typo | xlang |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BM25F only (no boosts, no prefix/typo/translit) | 82.2% | 87.7% | 0.857 | 80% | 0.10 | 0.19 | 71% | 67% | 100% | 100% | 100% | 20% | 40% | 95% |
| BM25F + prefix/typo/translit | 91.8% | 97.3% | 0.944 | 80% | 0.23 | 0.47 | 79% | 67% | 100% | 100% | 100% | 80% | 80% | 100% |
| Ours: + curated aliases, pins, boosts | 98.6% | 100.0% | 0.993 | 80% | 0.35 | 0.76 | 100% | 100% | 100% | 100% | 100% | 100% | 80% | 100% |
