# 500k ablation study

| Variant | PR-AUC | Recall | Precision | F1 | FPR | Cost / transaction | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.2337 | 0.8330 | 0.0764 | 0.1400 | 0.5145 | 17.91 | 12.13 |
| RulesOnly | 0.0924 | 1.0000 | 0.0486 | 0.0927 | 1.0000 | 19.03 | 2.06 |
| IsolationForest | 0.0707 | 0.9997 | 0.0487 | 0.0929 | 0.9976 | 19.00 | 9.35 |
| Hybrid_RulesSupervised | 0.2388 | 0.8829 | 0.0670 | 0.1246 | 0.6284 | 17.65 | 13.05 |
| Hybrid_AddAnomaly | 0.2392 | 0.8925 | 0.0650 | 0.1212 | 0.6559 | 17.71 | 61.71 |
| Hybrid_AddDeviceRecipient | 0.2324 | 0.8865 | 0.0663 | 0.1233 | 0.6386 | 17.67 | 23.17 |
| FullHybrid | 0.2318 | 0.8933 | 0.0655 | 0.1220 | 0.6515 | 17.58 | 20.24 |
| Hybrid_WithoutAnomaly | 0.2275 | 0.8536 | 0.0721 | 0.1330 | 0.5613 | 17.80 | 12.05 |
| Hybrid_WithoutGraph | 0.2283 | 0.8994 | 0.0644 | 0.1202 | 0.6678 | 17.60 | 21.25 |
| Hybrid_WithoutSequence | 0.2421 | 0.8722 | 0.0689 | 0.1277 | 0.6024 | 17.67 | 20.25 |
| WithoutGraphFeatures | 0.2319 | 0.8928 | 0.0647 | 0.1207 | 0.6593 | 17.76 | 11.24 |
| WithoutSequenceFeatures | 0.2282 | 0.9134 | 0.0620 | 0.1162 | 0.7060 | 17.65 | 10.82 |

Thresholds use validation-selected simulated minimum cost. Confidence intervals and reference models are in ablation_all_sizes.csv.
