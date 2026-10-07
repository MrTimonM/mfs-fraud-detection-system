# Common final test: error analysis

Synthetic data only. Errors are at each model's validation-selected STEP_UP threshold (max validation F1). Counts are descriptive, not causal.

## fraud_type

| Value | LightGBM_200k recall / FPR / FN / FP | Hybrid_WithoutSequence_500k recall / FPR / FN / FP | XGBoost_500k recall / FPR / FN / FP | FullHybrid_500k recall / FPR / FN / FP |
|---|---:|---:|---:|---:|
| ACCOUNT_TAKEOVER | 0.659 / — / 326 / 0 | 0.674 / — / 312 / 0 | 0.666 / — / 320 / 0 | 0.678 / — / 308 / 0 |
| AGENT_FRAUD | 0.016 / — / 304 / 0 | 0.016 / — / 304 / 0 | 0.010 / — / 306 / 0 | 0.016 / — / 304 / 0 |
| CASH_OUT_ABUSE | 0.218 / — / 284 / 0 | 0.226 / — / 281 / 0 | 0.201 / — / 290 / 0 | 0.242 / — / 275 / 0 |
| DEVICE_FRAUD | 0.264 / — / 393 / 0 | 0.251 / — / 400 / 0 | 0.258 / — / 396 / 0 | 0.266 / — / 392 / 0 |
| LEGITIMATE | — / 0.027 / 0 / 2586 | — / 0.029 / 0 / 2732 | — / 0.026 / 0 / 2462 | — / 0.031 / 0 / 2935 |
| MULE_ACTIVITY | 0.387 / — / 390 / 0 | 0.420 / — / 369 / 0 | 0.330 / — / 426 / 0 | 0.454 / — / 347 / 0 |
| SIM_SWAP_PATTERN | 0.550 / — / 143 / 0 | 0.563 / — / 139 / 0 | 0.544 / — / 145 / 0 | 0.572 / — / 136 / 0 |
| SOCIAL_ENGINEERING | 0.384 / — / 362 / 0 | 0.420 / — / 341 / 0 | 0.374 / — / 368 / 0 | 0.430 / — / 335 / 0 |
| SUBTLE_FRAUD | 0.010 / — / 897 / 0 | 0.012 / — / 895 / 0 | 0.007 / — / 900 / 0 | 0.014 / — / 893 / 0 |
| VELOCITY_FRAUD | 0.010 / — / 606 / 0 | 0.010 / — / 606 / 0 | 0.008 / — / 607 / 0 | 0.013 / — / 604 / 0 |

## transaction_type

| Value | LightGBM_200k recall / FPR / FN / FP | Hybrid_WithoutSequence_500k recall / FPR / FN / FP | XGBoost_500k recall / FPR / FN / FP | FullHybrid_500k recall / FPR / FN / FP |
|---|---:|---:|---:|---:|
| BANK_TRANSFER | 0.030 / 0.004 / 193 / 23 | 0.035 / 0.003 / 192 / 14 | 0.035 / 0.003 / 192 / 16 | 0.035 / 0.003 / 192 / 15 |
| BILL_PAYMENT | 0.043 / 0.004 / 178 / 23 | 0.043 / 0.003 / 178 / 15 | 0.043 / 0.003 / 178 / 18 | 0.043 / 0.003 / 178 / 18 |
| CASH_IN | 0.046 / 0.003 / 608 / 55 | 0.039 / 0.002 / 612 / 41 | 0.041 / 0.002 / 611 / 46 | 0.044 / 0.003 / 609 / 51 |
| CASH_OUT | 0.123 / 0.020 / 839 / 328 | 0.126 / 0.024 / 836 / 390 | 0.115 / 0.020 / 847 / 335 | 0.132 / 0.026 / 831 / 422 |
| MERCHANT_PAYMENT | 0.053 / 0.003 / 267 / 30 | 0.039 / 0.003 / 271 / 25 | 0.043 / 0.004 / 270 / 33 | 0.039 / 0.003 / 271 / 32 |
| MOBILE_RECHARGE | 0.061 / 0.003 / 214 / 23 | 0.039 / 0.003 / 219 / 21 | 0.044 / 0.003 / 218 / 21 | 0.044 / 0.003 / 218 / 23 |
| REMITTANCE | 0.054 / 0.002 / 141 / 11 | 0.020 / 0.002 / 146 / 11 | 0.027 / 0.003 / 145 / 12 | 0.027 / 0.003 / 145 / 13 |
| SEND_MONEY | 0.511 / 0.076 / 1265 / 2093 | 0.538 / 0.081 / 1193 / 2215 | 0.498 / 0.072 / 1297 / 1981 | 0.555 / 0.086 / 1150 / 2361 |

## channel

| Value | LightGBM_200k recall / FPR / FN / FP | Hybrid_WithoutSequence_500k recall / FPR / FN / FP | XGBoost_500k recall / FPR / FN / FP | FullHybrid_500k recall / FPR / FN / FP |
|---|---:|---:|---:|---:|
| AGENT | 0.219 / 0.024 / 1332 / 739 | 0.229 / 0.027 / 1314 / 818 | 0.208 / 0.024 / 1350 / 714 | 0.234 / 0.029 / 1306 / 884 |
| API | 0.409 / 0.056 / 178 / 195 | 0.402 / 0.055 / 180 / 191 | 0.385 / 0.049 / 185 / 171 | 0.415 / 0.058 / 176 / 201 |
| APP | 0.298 / 0.025 / 1305 / 919 | 0.302 / 0.026 / 1298 / 947 | 0.285 / 0.023 / 1329 / 869 | 0.317 / 0.027 / 1271 / 1014 |
| USSD | 0.321 / 0.026 / 718 / 529 | 0.338 / 0.027 / 700 / 553 | 0.316 / 0.024 / 724 / 496 | 0.349 / 0.029 / 689 / 597 |
| WEB | 0.425 / 0.058 / 172 / 204 | 0.482 / 0.063 / 155 / 223 | 0.431 / 0.060 / 170 / 212 | 0.492 / 0.068 / 152 / 239 |

## amount_range

| Value | LightGBM_200k recall / FPR / FN / FP | Hybrid_WithoutSequence_500k recall / FPR / FN / FP | XGBoost_500k recall / FPR / FN / FP | FullHybrid_500k recall / FPR / FN / FP |
|---|---:|---:|---:|---:|
| 2k-10k | 0.333 / 0.033 / 1561 / 1360 | 0.345 / 0.035 / 1532 / 1458 | 0.321 / 0.032 / 1589 / 1322 | 0.354 / 0.037 / 1512 / 1540 |
| 500-2k | 0.220 / 0.019 / 1574 / 743 | 0.230 / 0.019 / 1554 / 765 | 0.213 / 0.017 / 1589 / 671 | 0.242 / 0.022 / 1530 / 858 |
| <=500 | 0.170 / 0.017 / 381 / 145 | 0.163 / 0.016 / 384 / 138 | 0.142 / 0.013 / 394 / 115 | 0.176 / 0.018 / 378 / 153 |
| >10k | 0.534 / 0.064 / 189 / 338 | 0.564 / 0.070 / 177 / 371 | 0.542 / 0.067 / 186 / 354 | 0.571 / 0.073 / 174 / 384 |

## device

| Value | LightGBM_200k recall / FPR / FN / FP | Hybrid_WithoutSequence_500k recall / FPR / FN / FP | XGBoost_500k recall / FPR / FN / FP | FullHybrid_500k recall / FPR / FN / FP |
|---|---:|---:|---:|---:|
| known_device | 0.188 / 0.016 / 3279 / 1419 | 0.198 / 0.017 / 3240 / 1486 | 0.170 / 0.014 / 3351 / 1204 | 0.207 / 0.019 / 3201 / 1621 |
| new_device | 0.640 / 0.154 / 426 / 1167 | 0.656 / 0.164 / 407 / 1246 | 0.656 / 0.166 / 407 / 1258 | 0.668 / 0.173 / 393 / 1314 |

## recipient

| Value | LightGBM_200k recall / FPR / FN / FP | Hybrid_WithoutSequence_500k recall / FPR / FN / FP | XGBoost_500k recall / FPR / FN / FP | FullHybrid_500k recall / FPR / FN / FP |
|---|---:|---:|---:|---:|
| first_time_recipient | 0.387 / 0.074 / 1752 / 1932 | 0.423 / 0.083 / 1650 / 2171 | 0.386 / 0.074 / 1756 / 1931 | 0.436 / 0.089 / 1612 / 2334 |
| known_recipient | 0.174 / 0.010 / 1953 / 654 | 0.156 / 0.008 / 1997 / 561 | 0.153 / 0.008 / 2002 / 531 | 0.162 / 0.009 / 1982 / 601 |

## customer_history

| Value | LightGBM_200k recall / FPR / FN / FP | Hybrid_WithoutSequence_500k recall / FPR / FN / FP | XGBoost_500k recall / FPR / FN / FP | FullHybrid_500k recall / FPR / FN / FP |
|---|---:|---:|---:|---:|
| cold_start_lt5_events | 0.204 / 0.017 / 249 / 98 | 0.233 / 0.020 / 240 / 118 | 0.224 / 0.019 / 243 / 112 | 0.240 / 0.021 / 238 / 123 |
| warm_history | 0.296 / 0.028 / 3456 / 2488 | 0.306 / 0.029 / 3407 / 2614 | 0.284 / 0.026 / 3515 / 2350 | 0.316 / 0.032 / 3356 / 2812 |

## account_age

| Value | LightGBM_200k recall / FPR / FN / FP | Hybrid_WithoutSequence_500k recall / FPR / FN / FP | XGBoost_500k recall / FPR / FN / FP | FullHybrid_500k recall / FPR / FN / FP |
|---|---:|---:|---:|---:|
| 1-2.7y | 0.304 / 0.026 / 950 / 645 | 0.319 / 0.028 / 930 / 698 | 0.294 / 0.025 / 964 / 627 | 0.325 / 0.030 / 921 / 754 |
| 91-365d | 0.280 / 0.027 / 471 / 309 | 0.289 / 0.030 / 465 / 341 | 0.266 / 0.028 / 480 / 320 | 0.306 / 0.032 / 454 / 364 |
| <=90d | 0.250 / 0.024 / 117 / 59 | 0.276 / 0.027 / 113 / 65 | 0.269 / 0.025 / 114 / 61 | 0.288 / 0.031 / 111 / 75 |
| >2.7y | 0.289 / 0.028 / 2167 / 1573 | 0.298 / 0.029 / 2139 / 1628 | 0.278 / 0.026 / 2200 / 1454 | 0.308 / 0.031 / 2108 / 1742 |

## night

| Value | LightGBM_200k recall / FPR / FN / FP | Hybrid_WithoutSequence_500k recall / FPR / FN / FP | XGBoost_500k recall / FPR / FN / FP | FullHybrid_500k recall / FPR / FN / FP |
|---|---:|---:|---:|---:|
| day | 0.284 / 0.028 / 3152 / 2167 | 0.296 / 0.029 / 3099 / 2273 | 0.275 / 0.026 / 3192 / 2048 | 0.307 / 0.031 / 3053 / 2454 |
| night | 0.324 / 0.026 / 553 / 419 | 0.330 / 0.028 / 548 / 459 | 0.308 / 0.025 / 566 / 414 | 0.339 / 0.029 / 541 / 481 |

## velocity

| Value | LightGBM_200k recall / FPR / FN / FP | Hybrid_WithoutSequence_500k recall / FPR / FN / FP | XGBoost_500k recall / FPR / FN / FP | FullHybrid_500k recall / FPR / FN / FP |
|---|---:|---:|---:|---:|
| high_velocity_ge2_in_5m | 0.260 / 0.033 / 433 / 300 | 0.284 / 0.038 / 419 / 345 | 0.265 / 0.035 / 430 / 313 | 0.294 / 0.040 / 413 / 366 |
| normal_velocity | 0.295 / 0.027 / 3272 / 2286 | 0.304 / 0.028 / 3228 / 2387 | 0.282 / 0.025 / 3328 / 2149 | 0.314 / 0.030 / 3181 / 2569 |

## graph_mule_risk

| Value | LightGBM_200k recall / FPR / FN / FP | Hybrid_WithoutSequence_500k recall / FPR / FN / FP | XGBoost_500k recall / FPR / FN / FP | FullHybrid_500k recall / FPR / FN / FP |
|---|---:|---:|---:|---:|
| high_>50 | 0.308 / 0.025 / 54 / 38 | 0.256 / 0.023 / 58 / 35 | 0.282 / 0.024 / 56 / 37 | 0.256 / 0.023 / 58 / 35 |
| low_<=20 | 0.290 / 0.027 / 3643 / 2539 | 0.302 / 0.029 / 3581 / 2682 | 0.280 / 0.026 / 3693 / 2417 | 0.312 / 0.031 / 3528 / 2882 |
| medium_20-50 | 0.467 / 0.080 / 8 / 9 | 0.467 / 0.134 / 8 / 15 | 0.400 / 0.071 / 9 / 8 | 0.467 / 0.161 / 8 / 18 |

## Observations

- Largest missed-fraud category for LightGBM_200k: **SUBTLE_FRAUD** (897 false negatives, recall 0.010).
- Segments carrying the largest share of false positives: graph_mule_risk=low_<=20 (98.2%); customer_history=warm_history (96.2%); velocity=normal_velocity (88.4%); night=day (83.8%).
- Lowest-recall segments: transaction_type=BANK_TRANSFER (recall 0.030, n fraud 199); transaction_type=BILL_PAYMENT (recall 0.043, n fraud 186); transaction_type=CASH_IN (recall 0.046, n fraud 637); transaction_type=MERCHANT_PAYMENT (recall 0.053, n fraud 282).
