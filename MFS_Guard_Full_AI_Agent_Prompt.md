# MFS Guard — Full AI Agent Prompt for Synthetic Dataset Generation, Model Training, Evaluation, and Scaling

## Role

You are a **senior machine learning engineer, fraud-detection researcher, data engineer, and MLOps engineer**.

Your task is to build a complete, reproducible, production-oriented experimentation pipeline for a Mobile Financial Services fraud-detection project called **MFS Guard**.

The system must generate realistic synthetic MFS transaction datasets at multiple scales, train multiple fraud-detection models, compare them rigorously, save all artifacts, and produce a final report showing how performance, training cost, latency, and scalability change as dataset size increases.

This is not a toy notebook exercise. Build it as a proper Python project with reusable modules, CLI commands, validation, logging, tests, saved datasets, model artifacts, plots, metrics, experiment summaries, and final comparison reports.

---

# 1. Main Goal

Build and evaluate a hybrid fraud-intelligence architecture that can eventually support real-time pre-transaction fraud screening.

The target architecture is:

```text
Incoming Transaction
        |
        v
Feature Engineering / Historical Context
        |
        +---------------------+
        |                     |
        v                     v
Deterministic Rules      Supervised ML
        |                     |
        +----------+----------+
                   |
                   v
          Behavioral Anomaly Model
                   |
                   v
        Device / Recipient / Graph Risk
                   |
                   v
           Sequence / Temporal Risk
                   |
                   v
             Risk Fusion Layer
                   |
                   v
     APPROVE / MONITOR / STEP_UP_AUTH
          / HOLD / REJECT
```

The project must demonstrate:

- supervised fraud classification,
- customer-specific behavioral anomaly detection,
- deterministic fraud rules,
- device-risk intelligence,
- recipient-risk intelligence,
- transaction-sequence intelligence,
- graph/mule-network intelligence,
- calibrated risk probabilities,
- explainability,
- human-review compatibility,
- business-cost-aware threshold selection,
- latency benchmarking,
- reproducible experiments.

---

# 2. Dataset Scaling Experiments

Generate and evaluate **five datasets**:

```text
100,000 transactions
200,000 transactions
300,000 transactions
400,000 transactions
500,000 transactions
```

For each dataset size, perform the entire workflow:

```text
generate
→ validate
→ engineer features
→ leakage checks
→ chronological split
→ train models
→ tune selected models
→ calibrate probabilities
→ evaluate
→ optimize threshold
→ run anomaly model
→ run rule engine
→ run hybrid/fusion model
→ calculate explainability
→ benchmark latency
→ benchmark training time
→ save models
→ save metrics
→ save predictions
→ save plots
→ save reports
```

After all five experiments finish:

```text
100k vs 200k vs 300k vs 400k vs 500k
```

Create a final comparison and recommendation.

---

# 3. Mandatory Backward-Compatible MFS Guard Columns

Every generated dataset MUST contain the following original MFS Guard columns.

Do not remove them.

Do not rename them.

```text
transaction_id
user_id
receiver_id
device_id
transaction_type
channel
amount
fee
balance_before
balance_after
depletion_ratio
failed_pin_attempts
otp_resend_count
pin_reset_recently
device_is_new
rooted_device
emulator_detected
vpn_active
screen_share_detected
channel_changed_recently
tx_count_5m
tx_count_15m
tx_count_1h
tx_count_24h
amount_sum_5m
amount_sum_1h
balance_inquiry_count_5m
turnaround_latency_seconds
latitude
longitude
impossible_travel
impossible_travel_speed
recipient_risk_score
device_risk_score
user_behavior_score
blacklisted_device
blacklisted_agent
rule_risk_score
timestamp
fraud_label
```

Additionally, add:

```text
fraud_type
```

`fraud_label` must be binary:

```text
0 = legitimate
1 = fraud
```

`fraud_type` must use:

```text
LEGITIMATE
ACCOUNT_TAKEOVER
MULE_ACTIVITY
VELOCITY_FRAUD
SUBTLE_FRAUD
DEVICE_FRAUD
SOCIAL_ENGINEERING
AGENT_FRAUD
CASH_OUT_ABUSE
SIM_SWAP_PATTERN
```

Not every fraud category needs equal frequency.

---

# 4. Full Recommended Dataset Schema

Include all mandatory original columns above, plus the following additional fields when technically meaningful.

## 4.1 Transaction identity and entity fields

```text
transaction_id
user_id
receiver_id
device_id
agent_id
merchant_id
session_id
```

## 4.2 Transaction context

```text
transaction_type
channel
amount
fee
balance_before
balance_after
depletion_ratio
currency
timestamp
transaction_hour
day_of_week
weekend_flag
is_night_transaction
```

Suggested transaction types:

```text
SEND_MONEY
CASH_OUT
CASH_IN
MERCHANT_PAYMENT
MOBILE_RECHARGE
BILL_PAYMENT
BANK_TRANSFER
REMITTANCE
```

Suggested channels:

```text
APP
USSD
AGENT
WEB
API
```

---

# 5. Customer-Level Profile Fields

Create persistent customers instead of generating unrelated rows.

Each user should have a behavioral profile.

Add or maintain internally:

```text
account_age_days
customer_segment
home_latitude
home_longitude
preferred_channel
preferred_transaction_type
typical_transaction_amount
typical_daily_transaction_count
normal_active_hour_start
normal_active_hour_end
normal_location_radius_km
usual_recipient_count
historical_device_count
wallet_balance_tendency
```

The generator should use these persistent profiles to produce realistic transaction histories.

---

# 6. Credential and Authentication Features

Include:

```text
failed_pin_attempts
otp_resend_count
pin_reset_recently
password_reset_recently
biometric_failed_recently
login_failures_1h
login_failures_24h
credential_change_count_24h
```

These fields must not perfectly determine fraud.

Legitimate users may occasionally reset credentials or fail authentication.

---

# 7. Device and Session Features

Include:

```text
device_is_new
device_first_seen
device_age_days
rooted_device
emulator_detected
vpn_active
screen_share_detected
sim_changed_recently
device_os_changed
device_fingerprint_changed
shared_device_user_count
user_device_count
device_transaction_count_24h
device_transaction_count_7d
device_risk_score
blacklisted_device
```

Device-risk scores must be generated from historical or contextual signals only.

Do not derive current transaction fraud labels into device-risk features.

---

# 8. Channel Features

Include:

```text
channel
channel_changed_recently
preferred_channel_mismatch
channel_switch_count_24h
```

Example suspicious sequence:

```text
APP balance inquiry
→ USSD high-value transfer
→ AGENT cash-out
```

---

# 9. Transaction Velocity Features

These are mandatory.

```text
tx_count_5m
tx_count_15m
tx_count_1h
tx_count_6h
tx_count_24h

amount_sum_5m
amount_sum_15m
amount_sum_1h
amount_sum_6h
amount_sum_24h

avg_amount_1h
avg_amount_24h
max_amount_24h

time_since_previous_tx_seconds
```

All rolling features must use only transactions occurring before the current transaction.

Never use future transactions.

---

# 10. Balance and Liquidity Features

Include:

```text
balance_before
balance_after
fee
depletion_ratio
balance_change_ratio
recent_inflow_amount_1h
recent_outflow_amount_1h
recent_inflow_amount_24h
recent_outflow_amount_24h
turnaround_latency_seconds
rapid_fund_turnaround
```

Calculate:

```text
depletion_ratio = (amount + fee) / max(balance_before, epsilon)
```

Do not generate depletion ratio independently.

It must be derived from the transaction values.

---

# 11. Balance Inquiry Features

Include:

```text
balance_inquiry_count_5m
balance_inquiry_count_15m
balance_inquiry_count_1h
balance_inquiry_then_transfer
```

Repeated balance inquiries followed by a high-value transaction should sometimes be suspicious but not always fraudulent.

---

# 12. Customer Behavioral Baseline Features

Include:

```text
user_mean_amount
user_median_amount
user_std_amount
user_max_amount_30d
user_mean_daily_tx_count
user_mean_hourly_tx_count

amount_vs_user_mean
amount_vs_user_median
amount_zscore_user
amount_percentile_user

unusual_hour_score
unusual_channel_score
unusual_location_score
unusual_transaction_type_score

user_behavior_score
```

All customer baseline statistics must use historical transactions only.

The current transaction must not contribute to its own baseline.

---

# 13. Recipient Relationship Features

Include:

```text
first_time_recipient
recipient_frequency_user
user_receiver_tx_count_7d
user_receiver_tx_count_30d
time_since_last_recipient_tx
receiver_account_age_days
receiver_transaction_count_24h
receiver_transaction_count_7d
receiver_unique_senders_24h
receiver_unique_senders_7d
receiver_unique_receivers_7d
receiver_inflow_1h
receiver_inflow_24h
receiver_outflow_1h
receiver_outflow_24h
receiver_turnaround_ratio
receiver_sender_diversity
recipient_risk_score
```

Recipient risk must represent known historical behavior only.

Do not derive it from current target labels.

---

# 14. Agent Features

Include where applicable:

```text
agent_id
agent_transaction_count_1h
agent_transaction_count_24h
agent_cashout_amount_1h
agent_cashout_amount_24h
agent_unique_users_24h
agent_risk_score
blacklisted_agent
```

Some transactions may have no agent.

Use safe missing values.

---

# 15. Merchant Features

Include where applicable:

```text
merchant_id
merchant_category
merchant_transaction_count_24h
merchant_avg_amount_30d
merchant_risk_score
```

Merchant fields may be null for non-merchant transactions.

---

# 16. Location Features

Keep the original fields:

```text
latitude
longitude
impossible_travel
impossible_travel_speed
```

Add:

```text
previous_latitude
previous_longitude
distance_from_previous_tx_km
time_since_previous_location_seconds
distance_from_home_km
location_first_seen
location_risk_score
```

Calculate:

```text
distance_from_previous_tx_km
```

using a geodesic or Haversine calculation.

Calculate:

```text
impossible_travel_speed
=
distance_from_previous_tx_km
/
time_since_previous_location_hours
```

Set:

```text
impossible_travel
```

based on a configurable realistic threshold.

Do not randomly generate impossible-travel speed independently.

---

# 17. Sequence / Temporal Fraud Features

Create interpretable sequence features:

```text
pin_reset_then_transfer
new_device_then_transfer
failed_pin_then_success
balance_inquiry_then_cashout
incoming_then_rapid_outgoing
new_recipient_then_large_transfer
multiple_otp_then_transfer
device_change_then_high_value
sim_change_then_transfer
credential_change_then_cashout
channel_hop_sequence
```

Generate:

```text
sequence_risk_score
```

This score must be based on the above historical sequences, not current labels.

---

# 18. Graph and Mule-Network Features

Create a directed transaction graph.

Nodes:

```text
users
wallets
recipients
```

Edges:

```text
sender → receiver
```

Derive efficient graph features such as:

```text
sender_in_degree
sender_out_degree
receiver_in_degree
receiver_out_degree
unique_sender_count
unique_receiver_count
pagerank
connected_component_size
reciprocal_transfer_ratio
fund_concentration
rapid_forwarding_ratio
shared_recipient_count
shared_device_count
mule_network_score
```

Graph features must be generated using only historical graph state before the current transaction or by safe periodic snapshots.

Do not leak future fraud information.

For 500k rows, optimize graph calculations so the process remains computationally reasonable.

---

# 19. Rule Engine

Implement a configurable deterministic rule engine.

Rules must not be scattered as hard-coded logic throughout the codebase.

Use a configuration structure.

At minimum include rules for:

```text
high balance depletion
critical balance depletion
failed PIN surge
OTP resend surge
recent PIN reset
new device
new device + high amount
rooted device
emulator
VPN
high transaction velocity
rapid fund turnaround
channel hopping
balance inquiry surge
impossible travel
high-risk recipient
blacklisted device
blacklisted agent
SIM change + high-value transaction
```

Calculate:

```text
rule_risk_score
```

on a 0–100 scale.

Save:

```text
triggered_rule_count
highest_rule_severity
```

as optional additional fields.

IMPORTANT:

`rule_risk_score` should NOT automatically be used as an input to every standalone supervised model.

Use it mainly for:

```text
Rule-only baseline
Hybrid/fusion model
```

This prevents ML models from simply reproducing the deterministic rule engine.

---

# 20. Fraud Generation Philosophy

Do NOT create fraud using simple perfectly deterministic rules.

Bad example:

```python
if device_is_new == 1 and amount > 30000:
    fraud_label = 1
```

Do not do this.

Fraud generation must be latent-event-based, probabilistic, noisy, and overlapping.

---

# 21. Fraud Scenario Generation

## 21.1 Account Takeover

Possible signals:

```text
new or unusual device
unusual login location
credential failures
OTP resend activity
PIN reset
new recipient
unusual amount
unusual transaction time
channel change
```

Not every account takeover must have every signal.

Some legitimate transactions must also contain some of these signals.

---

## 21.2 Mule Activity

Possible behavior:

```text
many senders → one receiver
rapid inflow
rapid onward transfer
rapid cash-out
shared devices
recipient network clustering
high sender diversity
high turnover
```

---

## 21.3 Velocity Fraud

Possible behavior:

```text
many transactions in short period
repeated transfers
amount splitting
burst activity
rapid recipient switching
```

---

## 21.4 Subtle Fraud

Design intentionally difficult cases:

```text
amount close to normal
known device
normal location
slightly unusual recipient
gradual behavioral shift
low rule score
```

These cases are important for proving ML value beyond static rules.

---

## 21.5 Device Fraud

Possible signals:

```text
emulator
rooted device
shared device
new device
device fingerprint drift
VPN
screen-sharing
```

Do not make device flags perfectly predictive.

---

## 21.6 Social Engineering

Possible characteristics:

```text
legitimate authenticated device
new recipient
unusual transfer amount
unusual urgency
possibly normal credentials
customer voluntarily sends money
```

This category should be harder for pure device-security models.

---

## 21.7 Agent Fraud

Possible patterns:

```text
unusual cash-out volume
many customers through same agent
agent activity spike
high-risk device reuse
rapid cash-out concentration
```

---

## 21.8 Cash-Out Abuse

Possible behavior:

```text
large incoming amount
rapid cash-out
high depletion
unusual agent
repeated cash-out
```

---

## 21.9 SIM Swap Pattern

Possible behavior:

```text
recent SIM change
credential reset
new device
high-value transaction
new recipient
```

Do not make SIM change alone sufficient for fraud.

---

# 22. Dataset Realism Requirements

Generate persistent entities.

Do not generate unrelated transactions.

Each user must have:

```text
normal spending pattern
normal transaction frequency
typical active hours
normal location area
common recipients
usual devices
preferred channel
```

Transactions should span several simulated months.

Suggested period:

```text
90–180 days
```

Generate chronological timestamps.

---

# 23. Fraud Prevalence

Target approximately:

```text
4%–6% fraud
```

for the main benchmark.

Aim for approximately 5%.

Do not force exactly 5%.

Allow natural variation.

Save class distributions for every dataset.

---

# 24. Scaling Users With Dataset Size

Suggested user counts:

```text
100k dataset → ~10k–15k users
200k dataset → ~20k–25k users
300k dataset → ~30k–35k users
400k dataset → ~40k–45k users
500k dataset → ~50k+ users
```

However, preserve enough transaction history per user to support behavioral modeling.

If this scaling creates histories that are too short, use fewer users.

Behavioral realism is more important than hitting an arbitrary user count.

---

# 25. Critical Target Leakage Prevention

This is mandatory.

Before training, inspect every column for target leakage.

Never train the supervised models directly on:

```text
fraud_label
fraud_type
post-investigation status
analyst decision
future transaction information
future confirmed fraud counts
future blacklist assignments
```

Be especially careful with:

```text
recipient_risk_score
device_risk_score
user_behavior_score
rule_risk_score
blacklisted_device
blacklisted_agent
```

If a field was created using future or current fraud labels, it must not be used as a predictive input.

Historical risk features must only use information available before the current transaction timestamp.

Produce a leakage audit report.

Save:

```text
results/<size>/leakage_audit.json
```

---

# 26. Feature Sets for Fair Model Comparison

Create at least these feature groups.

## Feature Set A — Raw + Historical Behavioral

Exclude:

```text
rule_risk_score
fraud_label
fraud_type
```

Use for primary supervised ML comparison.

## Feature Set B — Behavioral Anomaly

Use primarily continuous customer-relative behavioral features for Isolation Forest.

## Feature Set C — Rule Engine

Use deterministic rule signals only.

## Feature Set D — Hybrid Fusion

Inputs may include:

```text
supervised_probability
anomaly_score
rule_risk_score
recipient_risk_score
device_risk_score
sequence_risk_score
mule_network_score
```

provided every component is leakage-safe.

---

# 27. Dataset Validation

For every generated dataset verify:

```text
exact requested row count
unique transaction IDs
valid timestamps
chronological ordering
valid transaction amounts
no invalid negative balances unless intentionally simulated
valid categorical values
fraud prevalence
fraud-type distribution
duplicate rows
missing critical columns
NaN counts
infinite values
extreme outliers
feature consistency
balance equation consistency
rolling-window consistency
location consistency
```

Check:

```text
balance_after ≈ balance_before - amount - fee
```

where applicable for outgoing transactions.

Save:

```text
results/<size>/dataset_validation.json
```

If critical validation fails, stop the experiment and fix the generator.

---

# 28. Dataset Cards

Create:

```text
reports/dataset_cards/dataset_100k.md
reports/dataset_cards/dataset_200k.md
reports/dataset_cards/dataset_300k.md
reports/dataset_cards/dataset_400k.md
reports/dataset_cards/dataset_500k.md
```

Each must include:

```text
dataset size
user count
receiver count
device count
agent count
merchant count
date range
feature count
fraud prevalence
fraud-type distribution
transaction-type distribution
channel distribution
missing-value statistics
generation methodology
feature descriptions
fraud injection methodology
known limitations
leakage-prevention strategy
random seed
```

---

# 29. Data Storage

Save raw generated datasets as:

```text
data/raw/mfs_100k.csv
data/raw/mfs_200k.csv
data/raw/mfs_300k.csv
data/raw/mfs_400k.csv
data/raw/mfs_500k.csv
```

Also save Parquet versions for faster loading:

```text
data/raw/mfs_100k.parquet
...
```

Processed feature datasets should be stored separately:

```text
data/processed/
```

Do not overwrite raw datasets during preprocessing.

---

# 30. Reproducibility

Use:

```text
RANDOM_SEED = 42
```

All random generators must be seeded.

Store metadata:

```text
dataset_size
seed
generation_timestamp
Python version
package versions
feature list
fraud distribution
generation configuration
```

in:

```text
data/metadata/
```

---

# 31. Data Splitting Strategy

Primary evaluation must be chronological.

Use:

```text
Train      earliest 70%
Validation next 15%
Test       latest 15%
```

Do not use the latest test transactions while training or tuning.

Where feasible, also analyze user leakage across splits.

Save split time boundaries.

Do not use random train/test split as the primary result.

Optional:

perform a secondary random split only for comparison, clearly labeled as secondary.

---

# 32. Cold-Start Handling

Behavioral models must abstain when a user has insufficient history.

Example:

```text
history_count < configurable threshold
→ behavioral_model_status = INSUFFICIENT_HISTORY
```

Do not generate a fake confident anomaly score.

Fallback layers can still operate:

```text
Rule engine
Supervised classifier
Device risk
Recipient risk
```

Record:

```text
behavioral_history_count
behavioral_model_active
```

if useful.

---

# 33. Preprocessing

Use sklearn-compatible pipelines where possible.

Handle:

```text
numeric missing values
categorical missing values
categorical encoding
numeric scaling when required
```

Rules:

- fit preprocessors only on training data,
- never fit transformations using validation/test distributions,
- tree models do not need unnecessary scaling,
- Logistic Regression may require scaling,
- keep preprocessing reproducible.

---

# 34. Required Models

Train the following supervised models for every dataset size:

```text
Logistic Regression
Random Forest
HistGradientBoostingClassifier
XGBoost
LightGBM
```

Optional if available:

```text
CatBoost
```

Behavioral anomaly model:

```text
Isolation Forest
```

Optional anomaly models:

```text
Local Outlier Factor
One-Class SVM
Autoencoder
```

Only add optional models if computationally reasonable.

---

# 35. Class Imbalance Strategy

Do not optimize for accuracy.

Use:

```text
class_weight
scale_pos_weight
model-native imbalance handling
```

SMOTE is optional.

If used:

```text
apply only to training data
never validation
never test
```

Document which imbalance technique is used by each model.

---

# 36. Hyperparameter Tuning

Do not run huge exhaustive grids.

Use:

```text
RandomizedSearchCV
or
Optuna
```

Focus tuning on:

```text
LightGBM
XGBoost
Random Forest
```

Use a controlled compute budget.

Suggested approach:

```text
100k:
moderate tuning

200k:
reuse ranges and moderate tuning

300k–500k:
reuse best search space and reduce trials if necessary
```

Record:

```text
best parameters
best validation PR-AUC
tuning time
number of trials
```

---

# 37. Primary Evaluation Metric

Primary model-selection metric:

```text
PR-AUC / Average Precision
```

Do not choose a winner using accuracy alone.

Secondary metrics:

```text
precision
recall
F1
ROC-AUC
MCC
balanced accuracy
false positive rate
false negative rate
```

---

# 38. Required Binary Classification Metrics

For every model save:

```text
accuracy
balanced_accuracy
precision
recall
f1
roc_auc
pr_auc
average_precision
mcc
false_positive_rate
false_negative_rate
true_positive
false_positive
true_negative
false_negative
```

Also save:

```text
training_time_seconds
inference_time_seconds
model_file_size_mb
peak_memory_mb
```

where available.

---

# 39. Fraud-Type Classification

Build a second supervised task for fraud subtype analysis.

Task:

```text
LEGITIMATE
ACCOUNT_TAKEOVER
MULE_ACTIVITY
VELOCITY_FRAUD
SUBTLE_FRAUD
DEVICE_FRAUD
SOCIAL_ENGINEERING
AGENT_FRAUD
CASH_OUT_ABUSE
SIM_SWAP_PATTERN
```

At minimum test:

```text
LightGBM
XGBoost
Random Forest
```

Evaluate:

```text
macro F1
weighted F1
per-class precision
per-class recall
per-class F1
multiclass confusion matrix
```

This is secondary to binary fraud detection but should be included in the final analysis.

---

# 40. Confusion Matrices

Save for every supervised model:

```text
results/<size>/confusion_matrix_<model>.png
```

Also save raw confusion matrix values.

---

# 41. ROC Curves

Generate:

```text
individual ROC curves
combined ROC comparison
```

Save:

```text
results/<size>/roc_comparison.png
```

---

# 42. Precision-Recall Curves

Generate:

```text
combined precision-recall comparison
```

Save:

```text
results/<size>/pr_curve_comparison.png
```

This is one of the most important plots.

---

# 43. Threshold Optimization

Do not assume:

```text
threshold = 0.50
```

Evaluate a broad threshold range such as:

```text
0.05 → 0.95
```

For each threshold calculate:

```text
precision
recall
F1
false positives
false negatives
review volume
estimated business cost
```

Save:

```text
results/<size>/threshold_metrics.csv
results/<size>/threshold_curve.png
```

---

# 44. Multi-Action Decision Thresholds

Design configurable operational bands.

Example initial structure:

```text
0.00–0.25 → APPROVE
0.25–0.50 → APPROVE_AND_MONITOR
0.50–0.70 → STEP_UP_AUTH
0.70–0.90 → TEMPORARY_HOLD
0.90–1.00 → REJECT_OR_ANALYST_REVIEW
```

Do not treat these example thresholds as final.

Optimize them based on validation results and business-cost assumptions.

---

# 45. Business Cost Simulation

Add configurable assumptions to `config.yaml`.

Example categories:

```text
average_fraud_loss
false_positive_investigation_cost
customer_friction_cost
step_up_authentication_cost
temporary_hold_cost
```

Clearly label them as simulation assumptions.

Never claim they are actual upay figures.

Calculate:

```text
expected_business_cost
```

Example:

```text
false_negatives × average_fraud_loss
+ false_positives × false_positive_cost
+ step_up_count × verification_cost
+ temporary_holds × hold_cost
```

Produce:

```text
business_threshold_analysis.csv
```

---

# 46. Probability Calibration

For top supervised models evaluate:

```text
uncalibrated probability
Platt / sigmoid calibration
isotonic calibration
```

Compare:

```text
Brier score
calibration curve
PR-AUC
ROC-AUC
```

Select a calibrated version if it improves probability reliability without unacceptable discrimination loss.

Save:

```text
calibration_curve.png
calibration_metrics.csv
```

---

# 47. Isolation Forest

Train Isolation Forest primarily on legitimate historical behavior or an appropriate contamination-aware training subset.

Use customer-relative behavioral features.

Do not simply feed raw IDs.

Produce:

```text
anomaly_score
anomaly_flag
```

Evaluate its ability to detect:

```text
known fraud
subtle fraud
unseen fraud combinations
```

Also evaluate false-positive rate on legitimate transactions.

---

# 48. Rule-Only Baseline

Evaluate the deterministic rule engine as its own model.

Report:

```text
precision
recall
F1
PR-AUC if a continuous rule score exists
false positives
false negatives
```

This gives a baseline for answering:

```text
Does ML add value beyond rules?
```

---

# 49. Hybrid Risk Fusion

Build at least one learned fusion model.

Potential fusion inputs:

```text
supervised_fraud_probability
isolation_forest_anomaly_score
rule_risk_score
recipient_risk_score
device_risk_score
sequence_risk_score
mule_network_score
```

Recommended initial meta-model:

```text
Logistic Regression
```

Also optionally compare:

```text
LightGBM meta-model
```

Train the fusion model using validation-safe methodology.

Do not train and evaluate fusion on the same predictions.

Use out-of-fold or properly separated predictions where needed.

---

# 50. Ablation Study

For each useful dataset size, and especially the final 500k dataset, compare:

```text
Rules only
Isolation Forest only
Best supervised model only
Rules + supervised
Rules + supervised + anomaly
Rules + supervised + anomaly + device/recipient
Full hybrid including graph + sequence
```

Create an ablation table:

```text
System Variant
PR-AUC
Recall
Precision
F1
False Positive Rate
Expected Business Cost
Inference Latency
```

This should prove whether each layer provides value.

---

# 51. Unseen Fraud Experiment

Create a robustness experiment.

Example:

Train on most fraud categories while reducing or withholding selected fraud-pattern combinations.

Then test how well:

```text
supervised model
Isolation Forest
rule engine
hybrid system
```

detect unusual fraud scenarios.

Do not make claims about zero-day fraud beyond what the experiment actually demonstrates.

Clearly call this:

```text
synthetic unseen-pattern robustness experiment
```

---

# 52. Explainability

For tree-based supervised models, use SHAP where feasible.

For each dataset:

save:

```text
SHAP summary plot
global feature importance
top feature table
```

For selected suspicious transactions save local explanations:

```text
top_positive_risk_factors
top_negative_risk_factors
```

Example:

```text
New device
Amount 4.8× user historical mean
First-time recipient
6 transactions in 10 minutes
Recent PIN reset
Recipient has high sender diversity
```

Do not allow an LLM to invent explanations.

Any natural-language explanation must be grounded in structured evidence.

---

# 53. Feature Importance

Compare:

```text
LightGBM importance
XGBoost importance
Random Forest importance
SHAP importance
```

Create:

```text
results/<size>/feature_importance.csv
results/<size>/feature_importance.png
```

---

# 54. Latency Benchmarking

Benchmark end-to-end inference.

Measure:

```text
feature preparation latency
rule engine latency
supervised inference latency
anomaly inference latency
fusion latency
total inference latency
```

Report:

```text
mean
median
P50
P95
P99
```

Run enough requests for meaningful results.

If possible, test batch sizes:

```text
1
10
100
1000
```

Save:

```text
latency_benchmark.json
latency_benchmark.csv
```

---

# 55. Throughput Benchmarking

If feasible, estimate:

```text
transactions per second
```

for the trained inference pipeline.

Clearly state whether this is:

```text
single-process local benchmark
```

and do not imply production-scale throughput if it was not tested.

---

# 56. Training Resource Benchmarking

For every dataset/model combination record:

```text
training duration
CPU utilization where feasible
peak RAM
model size
dataset loading time
feature engineering time
```

This will help determine whether larger datasets provide enough benefit to justify their cost.

---

# 57. Model Persistence

Save trained models using:

```text
joblib
```

or the model library's recommended native serialization.

Directory:

```text
models/100k/
models/200k/
models/300k/
models/400k/
models/500k/
```

Each model directory should include:

```text
model file
preprocessor
feature list
best parameters
threshold
calibration model if applicable
metadata.json
```

---

# 58. Predictions

For every final test set save:

```text
transaction_id
fraud_label
fraud_type
predicted_probability
predicted_label
selected_threshold
model_name
```

For hybrid system add:

```text
rule_risk_score
anomaly_score
supervised_probability
final_risk_probability
final_action
```

Save:

```text
results/<size>/test_predictions.csv
```

---

# 59. Error Analysis

For each dataset and top model, inspect:

```text
false positives
false negatives
true positives
```

Group errors by:

```text
fraud_type
transaction_type
channel
amount band
device status
recipient status
customer history length
time of day
```

Create:

```text
error_analysis.md
error_analysis.csv
```

Identify patterns such as:

```text
subtle fraud frequently missed
new users generating false positives
social engineering difficult to detect
mule activity improved by graph features
```

Only report findings supported by results.

---

# 60. Fairness / Segment Analysis

Because this is synthetic data, do not make claims about real demographic fairness.

However, test consistency across synthetic operational segments:

```text
customer segment
account age group
transaction amount band
channel
region
device type
```

Compare:

```text
recall
false positive rate
```

Label this as:

```text
synthetic segment consistency analysis
```

---

# 61. Drift Simulation

If practical, simulate behavior drift between training and later test periods.

Examples:

```text
transaction amounts gradually increase
channel adoption shifts
new device types become common
fraud tactics become less obvious
recipient network patterns change
```

Measure performance degradation.

Calculate where appropriate:

```text
PSI
KS statistic
prediction distribution shift
```

Save drift reports.

---

# 62. Security / Adversarial Fraud Tests

Create adversarial synthetic scenarios such as:

```text
amount just below rule threshold
transaction splitting
slow fraud instead of bursts
trusted device fraud
small repeated transfers
mule chains
known-device account takeover
normal-location fraud
```

Measure:

```text
rule detection
supervised detection
anomaly detection
hybrid detection
```

Create:

```text
adversarial_test_results.csv
```

---

# 63. Required Plots

For each dataset generate at least:

```text
class distribution
fraud-type distribution
transaction amount distribution
fraud vs legitimate amount distribution
ROC comparison
PR comparison
confusion matrices
feature importance
SHAP summary
threshold curve
calibration curve
training time comparison
inference latency comparison
```

Final report should also include cross-dataset plots:

```text
dataset size vs PR-AUC
dataset size vs recall
dataset size vs precision
dataset size vs F1
dataset size vs training time
dataset size vs inference latency
dataset size vs model size
```

---

# 64. Final Cross-Dataset Comparison

After all experiments finish, create:

```text
results/final/all_model_results.csv
results/final/best_model_by_dataset.csv
results/final/dataset_size_comparison.csv
results/final/final_recommendation.md
```

Comparison columns should include:

```text
dataset_size
model
PR_AUC
ROC_AUC
precision
recall
F1
MCC
FPR
FNR
training_time
P95_inference_latency
model_size_mb
expected_business_cost
```

---

# 65. Determine Diminishing Returns

Analyze whether increasing dataset size materially improves model quality.

Example question:

```text
Does 500k significantly outperform 300k?
```

Do not automatically recommend the largest dataset.

Consider:

```text
performance gain
training cost
memory cost
latency
stability
```

The final recommendation may legitimately conclude that a smaller dataset offers nearly the same performance.

---

# 66. Required Project Structure

Create:

```text
mfs-guard-ml/
│
├── README.md
├── requirements.txt
├── config.yaml
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_generator.py
│   ├── entity_generator.py
│   ├── fraud_scenarios.py
│   ├── data_validation.py
│   ├── feature_engineering.py
│   ├── behavioral_features.py
│   ├── graph_features.py
│   ├── sequence_features.py
│   ├── rule_engine.py
│   ├── leakage_audit.py
│   ├── split_data.py
│   ├── preprocessing.py
│   ├── train_supervised.py
│   ├── train_anomaly.py
│   ├── calibration.py
│   ├── threshold_optimizer.py
│   ├── risk_fusion.py
│   ├── evaluate.py
│   ├── explain.py
│   ├── error_analysis.py
│   ├── drift_analysis.py
│   ├── adversarial_tests.py
│   ├── latency_benchmark.py
│   ├── resource_benchmark.py
│   ├── experiment_runner.py
│   └── final_comparison.py
│
├── models/
│   ├── 100k/
│   ├── 200k/
│   ├── 300k/
│   ├── 400k/
│   └── 500k/
│
├── results/
│   ├── 100k/
│   ├── 200k/
│   ├── 300k/
│   ├── 400k/
│   ├── 500k/
│   └── final/
│
├── reports/
│   ├── dataset_cards/
│   ├── experiment_reports/
│   └── final_report.md
│
├── scripts/
│   ├── generate_dataset.py
│   ├── generate_all.py
│   ├── train_experiment.py
│   ├── run_all_experiments.py
│   └── compare_all.py
│
└── tests/
    ├── test_generator.py
    ├── test_features.py
    ├── test_rules.py
    ├── test_leakage.py
    ├── test_splits.py
    └── test_metrics.py
```

---

# 67. Requirements File

Generate `requirements.txt`.

Include stable compatible versions where reasonable.

Required libraries:

```text
numpy
pandas
scipy
scikit-learn
xgboost
lightgbm
joblib
matplotlib
networkx
shap
pyarrow
pyyaml
tqdm
pytest
imbalanced-learn
psutil
```

Optional:

```text
optuna
memory-profiler
```

Do not depend on GPU libraries unless explicitly configured.

The project must work on CPU.

---

# 68. Logging

Use Python logging.

Every experiment should log:

```text
experiment start
dataset generation
feature engineering
validation
training start/end
tuning results
evaluation results
artifact paths
errors
experiment completion
```

Write logs to:

```text
logs/
```

---

# 69. Failure Handling

If one model fails:

```text
log the error
continue with remaining models
mark model status as FAILED
```

If dataset validation or leakage audit fails critically:

```text
stop that dataset experiment
```

Do not silently continue with corrupted data.

---

# 70. Memory Safety

500k transactions with many engineered features may consume significant memory.

Implement memory-aware processing.

Prefer:

```text
efficient dtypes
categorical dtypes
Parquet
vectorized operations
groupby/rolling optimization
incremental graph calculations
```

Avoid unnecessary full dataframe copies.

Delete temporary objects when appropriate.

---

# 71. Performance Constraints

Do not make the entire project dependent on extremely expensive algorithms.

The complete 100k–500k pipeline should be feasible on a modern consumer CPU workstation.

If a model or graph operation becomes prohibitively expensive:

```text
use an optimized approximation
document the reason
continue the benchmark
```

Do not skip the entire experiment.

---

# 72. README

Create a professional `README.md`.

Include:

```text
project overview
architecture
dataset design
mandatory schema
fraud categories
installation
commands
experiment workflow
models
metrics
folder structure
results interpretation
limitations
future integration plan
```

Mention explicitly:

```text
This project uses synthetic data only.
Results do not represent real upay production performance.
```

---

# 73. Full CLI Commands

The project must support commands similar to the following.

## Create virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

# 74. Dataset Generation Commands

The scripts must support:

```bash
python scripts/generate_dataset.py --rows 100000 --seed 42
python scripts/generate_dataset.py --rows 200000 --seed 42
python scripts/generate_dataset.py --rows 300000 --seed 42
python scripts/generate_dataset.py --rows 400000 --seed 42
python scripts/generate_dataset.py --rows 500000 --seed 42
```

Also:

```bash
python scripts/generate_all.py
```

`generate_all.py` must generate all five datasets sequentially.

---

# 75. Training Commands

Support:

```bash
python scripts/train_experiment.py --size 100k
python scripts/train_experiment.py --size 200k
python scripts/train_experiment.py --size 300k
python scripts/train_experiment.py --size 400k
python scripts/train_experiment.py --size 500k
```

Also:

```bash
python scripts/run_all_experiments.py
```

This command must:

```text
generate if missing
validate
feature-engineer
audit leakage
split
train
evaluate
save
```

for every dataset size.

---

# 76. Comparison Command

Support:

```bash
python scripts/compare_all.py
```

This must produce the final cross-dataset analysis.

---

# 77. Test Command

Support:

```bash
pytest -v
```

Tests should validate:

```text
generator output
mandatory columns
derived feature correctness
rule engine
leakage protection
chronological split
metric calculations
```

---

# 78. Optional Single Master Command

Also provide one master command:

```bash
python scripts/run_all_experiments.py --generate --train --evaluate --compare
```

If practical.

The script should clearly print progress such as:

```text
[1/5] Running 100k experiment
[2/5] Running 200k experiment
[3/5] Running 300k experiment
[4/5] Running 400k experiment
[5/5] Running 500k experiment
```

---

# 79. Experiment Reports

For every dataset create:

```text
reports/experiment_reports/experiment_100k.md
reports/experiment_reports/experiment_200k.md
reports/experiment_reports/experiment_300k.md
reports/experiment_reports/experiment_400k.md
reports/experiment_reports/experiment_500k.md
```

Include:

```text
dataset statistics
validation results
model metrics
winner
threshold
calibration
latency
training time
top features
error analysis
limitations
```

---

# 80. Final Report

Create:

```text
reports/final_report.md
```

Required sections:

```text
1. Executive Summary
2. Problem Definition
3. Dataset Generation Methodology
4. Dataset Schema
5. Fraud Scenarios
6. Leakage Prevention
7. Feature Engineering
8. Rule Engine
9. Supervised Models
10. Behavioral Anomaly Detection
11. Graph Intelligence
12. Sequence Intelligence
13. Risk Fusion
14. Evaluation Methodology
15. Results by Dataset Size
16. Model Comparison
17. Dataset Size Comparison
18. Ablation Study
19. Calibration
20. Threshold Optimization
21. Business Cost Analysis
22. Latency and Scalability
23. Error Analysis
24. Adversarial Testing
25. Synthetic Segment Analysis
26. Limitations
27. Recommended Final Architecture
28. Future Real-Data Validation Plan
29. Conclusion
```

---

# 81. Final Recommendation Requirements

At the end, explicitly answer:

```text
Which dataset size provided the best trade-off?

Which supervised model performed best?

Did the hybrid model beat standalone models?

Did Isolation Forest add measurable value?

Did graph features improve mule detection?

Did sequence features improve account-takeover detection?

What threshold should be used for the synthetic benchmark?

What are the main false-positive causes?

What are the main false-negative causes?

How much did 500k improve over 100k?

Was the 500k dataset worth its extra computation?

What should be deployed in the prototype demo?
```

---

# 82. Final Prototype Recommendation

The final recommended model stack should be determined from measured results, not assumptions.

A likely candidate architecture is:

```text
Rules
+
LightGBM or XGBoost
+
Isolation Forest
+
Device Risk
+
Recipient Risk
+
Graph Features
+
Sequence Risk
        ↓
Calibrated Fusion
        ↓
Risk Probability
        ↓
Operational Action
```

But do not declare this the winner until experiments are completed.

---

# 83. Important Scientific Rules

Follow these rules throughout:

1. Never optimize primarily for accuracy.
2. Never tune on the final test set.
3. Never calculate historical features using future transactions.
4. Never use `fraud_label` or `fraud_type` as model features.
5. Never hide failed experiments.
6. Never invent benchmark results.
7. Never claim synthetic performance equals production performance.
8. Never use arbitrary hybrid weights without validation.
9. Never create trivially separable fraud.
10. Never allow post-decision information into model training.
11. Always record assumptions.
12. Always save reproducible artifacts.
13. Always distinguish model probability from business decision.
14. Always preserve human-review capability for high-impact actions.

---

# 84. Code Quality Requirements

Code must be:

```text
modular
documented
typed where practical
PEP 8 compliant
reproducible
testable
memory-conscious
robust to missing values
```

Do not put the entire project in one Python file.

Avoid unnecessary duplication.

---

# 85. Mandatory Output Check

Before declaring the project complete, verify that these files exist:

```text
data/raw/mfs_100k.csv
data/raw/mfs_200k.csv
data/raw/mfs_300k.csv
data/raw/mfs_400k.csv
data/raw/mfs_500k.csv

results/100k/metrics.csv
results/200k/metrics.csv
results/300k/metrics.csv
results/400k/metrics.csv
results/500k/metrics.csv

results/final/all_model_results.csv
results/final/dataset_size_comparison.csv
results/final/final_recommendation.md

reports/dataset_cards/dataset_100k.md
reports/dataset_cards/dataset_200k.md
reports/dataset_cards/dataset_300k.md
reports/dataset_cards/dataset_400k.md
reports/dataset_cards/dataset_500k.md

reports/final_report.md
```

Also verify trained model artifacts exist for every successful experiment.

---

# 86. Execution Behavior

Work autonomously.

Do not merely give me code snippets.

Create the actual project files.

Run the scripts.

Fix errors encountered during execution.

Continue until all feasible experiments complete.

Do not stop after generating the 100k dataset.

Proceed sequentially:

```text
100k
→ train/evaluate/save

200k
→ train/evaluate/save

300k
→ train/evaluate/save

400k
→ train/evaluate/save

500k
→ train/evaluate/save

then final comparison
```

If computational limits prevent a specific expensive experiment, use a reasonable optimized alternative and document exactly what was changed and why.

---

# 87. Final Console Summary

When everything finishes, print a concise final summary similar to:

```text
=========================================================
MFS GUARD ML EXPERIMENTS COMPLETE
=========================================================

Datasets:
100k ✓
200k ✓
300k ✓
400k ✓
500k ✓

Best binary fraud model:
<model>

Best dataset size:
<size>

Best PR-AUC:
<value>

Recall:
<value>

Precision:
<value>

F1:
<value>

Best calibrated threshold:
<value>

Hybrid improvement over rules:
<value>

P95 inference latency:
<value>

Final report:
reports/final_report.md

Final comparison:
results/final/dataset_size_comparison.csv
=========================================================
```

Do not populate placeholder values until they have actually been measured.

---

# 88. Most Important Requirement

The purpose of this project is not to obtain artificially high accuracy.

The purpose is to build a credible fraud-detection research and prototype pipeline that demonstrates:

```text
realistic synthetic transaction behavior
customer-specific behavioral baselines
multiple fraud typologies
rules vs ML comparison
supervised fraud classification
anomaly detection
temporal intelligence
device intelligence
recipient intelligence
graph intelligence
explainability
responsible abstention
business-aware thresholding
real-time feasibility
scalability
reproducibility
```

Treat scientific credibility, leakage prevention, realistic evaluation, and explainability as more important than producing an unrealistically high score.

Start by creating the full project structure, configuration, requirements, dataset generator, validation pipeline, and the 100k experiment. After successfully validating the 100k workflow end-to-end, automatically continue with 200k, 300k, 400k, and 500k, then generate the final comparison and report.
