# MFS Guard common final test set (100k)

Synthetic data only. Results measured on it do not represent real upay production performance.

Generated with the unchanged training generator (`src/data_generator.py`), seed **2026** (never used for training, validation, tuning or tests; previously used seeds: [42, 123]). Histories are a self-contained 120-day chronological replay; every feature is computed from strictly earlier events of this set only. No training/validation row or transaction ID is reused.

## Summary

| Item | Value |
|---|---:|
| rows | 100000 |
| users | 1250 |
| receivers | 1250 |
| devices | 1666 |
| agents | 83 |
| merchants | 125 |
| columns | 154 |
| model_feature_count | 130 |
| fraud_count | 5223 |
| fraud_prevalence_percent | 5.223 |
| cold_start_percent | 6.25 |
| new_device_percent | 8.775 |
| new_recipient_percent | 28.977 |
| night_transaction_percent | 17.137 |
| high_velocity_percent | 9.642 |

Time range: 2026-01-01 01:40:13.018551464+00:00 to 2026-05-02 09:53:45.612996226+00:00

Definitions: cold start = fewer than five prior customer events (`behavioral_model_active == 0`, the anomaly layer abstains); new device = device not previously seen for that customer; new recipient = first transfer to that receiver; high velocity = two or more prior customer events in the previous five minutes.

## Fraud-type distribution

| Value | Count | % |
|---|---:|---:|
| LEGITIMATE | 94777 | 94.777 |
| ACCOUNT_TAKEOVER | 957 | 0.957 |
| SUBTLE_FRAUD | 906 | 0.906 |
| MULE_ACTIVITY | 636 | 0.636 |
| VELOCITY_FRAUD | 612 | 0.612 |
| SOCIAL_ENGINEERING | 588 | 0.588 |
| DEVICE_FRAUD | 534 | 0.534 |
| CASH_OUT_ABUSE | 363 | 0.363 |
| SIM_SWAP_PATTERN | 318 | 0.318 |
| AGENT_FRAUD | 309 | 0.309 |

## Channel distribution

| Value | Count | % |
|---|---:|---:|
| APP | 38937 | 38.937 |
| AGENT | 32011 | 32.011 |
| USSD | 21430 | 21.43 |
| WEB | 3828 | 3.828 |
| API | 3794 | 3.794 |

## Transaction-type distribution

| Value | Count | % |
|---|---:|---:|
| SEND_MONEY | 29969 | 29.969 |
| CASH_IN | 19577 | 19.577 |
| CASH_OUT | 17395 | 17.395 |
| MERCHANT_PAYMENT | 9587 | 9.587 |
| MOBILE_RECHARGE | 7393 | 7.393 |
| BILL_PAYMENT | 5849 | 5.849 |
| BANK_TRANSFER | 5658 | 5.658 |
| REMITTANCE | 4572 | 4.572 |

## Fraud recall-relevant slices (fraud rows)

| Slice | Fraud rows |
|---|---:|
| cold_start | 313 |
| new_device | 1184 |
| known_device | 4039 |
| first_time_recipient | 2858 |
| known_recipient | 2365 |
| high_velocity | 585 |
| sim_changed_recently | 269 |

## Missing values

Columns with any missing value: none. Total missing cells: 0. Absent agent/merchant identifiers are encoded as -1 by design.

## Validation

- Schema, accounting, chronology and sampled history recomputation: **PASSED** (96 sampled history checks)
- Leakage audit: **PASSED**; maximum absolute feature/target correlation 0.1509
- Transaction-ID overlap with all training datasets: **0**
- All supported fraud categories present: **True**

## Limitations

The population is new (different seed) but drawn from the same simulator as training, so this measures generalisation to fresh synthetic customers and episodes, not to real behaviour or new fraud mechanics. Unlike the per-scale test splits (last 15% of a 120-day replay with warm history), this set includes the cold-start warm-up period, so absolute metrics are not directly comparable with earlier per-scale results.
