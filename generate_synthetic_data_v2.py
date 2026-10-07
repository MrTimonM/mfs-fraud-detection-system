import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ============================================================
# CONFIG
# ============================================================

SEED = 42
N_ROWS = 100_000
TARGET_FRAUD_RATE = 0.06

rng = np.random.default_rng(SEED)

# ============================================================
# USER / DEVICE POPULATION
# ============================================================

N_USERS = 20_000
N_RECEIVERS = 12_000
N_DEVICES = 25_000

user_pool = np.array([
    f"USR{i:06d}"
    for i in range(1, N_USERS + 1)
])

receiver_pool = np.array([
    f"RCV{i:06d}"
    for i in range(1, N_RECEIVERS + 1)
])

device_pool = np.array([
    f"DEV{i:06d}"
    for i in range(1, N_DEVICES + 1)
])

# ============================================================
# USER BASE PROFILES
# ============================================================

# Different users naturally transact at different levels
user_typical_amount = rng.lognormal(
    mean=7.7,
    sigma=0.8,
    size=N_USERS
)

user_typical_amount = np.clip(
    user_typical_amount,
    100,
    30_000
)

user_typical_balance = rng.lognormal(
    mean=9.2,
    sigma=0.7,
    size=N_USERS
)

user_typical_balance = np.clip(
    user_typical_balance,
    1_000,
    250_000
)

# ============================================================
# TRANSACTION IDENTIFIERS
# ============================================================

transaction_id = np.array([
    f"TXN{i:07d}"
    for i in range(1, N_ROWS + 1)
])

user_index = rng.integers(
    0,
    N_USERS,
    N_ROWS
)

user_id = user_pool[user_index]

receiver_id = rng.choice(
    receiver_pool,
    size=N_ROWS
)

device_id = rng.choice(
    device_pool,
    size=N_ROWS
)

# ============================================================
# TIMESTAMP
# ============================================================

start_date = datetime.now() - timedelta(days=120)

seconds = rng.integers(
    0,
    120 * 24 * 60 * 60,
    N_ROWS
)

timestamp = np.array([
    start_date + timedelta(seconds=int(s))
    for s in seconds
])

# Time-related features
hour = np.array([
    t.hour
    for t in timestamp
])

day_of_week = np.array([
    t.weekday()
    for t in timestamp
])

is_night = (
    (hour <= 5) | (hour >= 23)
).astype(int)

# ============================================================
# TRANSACTION TYPE
# ============================================================

transaction_type = rng.choice(
    [
        "SEND_MONEY",
        "CASH_OUT",
        "PAYMENT",
        "MOBILE_RECHARGE"
    ],
    size=N_ROWS,
    p=[0.34, 0.27, 0.29, 0.10]
)

channel = rng.choice(
    ["APP", "USSD", "AGENT"],
    size=N_ROWS,
    p=[0.62, 0.18, 0.20]
)

# ============================================================
# BALANCE
# ============================================================

balance_noise = rng.lognormal(
    mean=0,
    sigma=0.35,
    size=N_ROWS
)

balance_before = (
    user_typical_balance[user_index]
    * balance_noise
)

balance_before = np.clip(
    balance_before,
    500,
    300_000
)

# ============================================================
# AMOUNT
# ============================================================

amount_noise = rng.lognormal(
    mean=0,
    sigma=0.65,
    size=N_ROWS
)

amount = (
    user_typical_amount[user_index]
    * amount_noise
)

# Occasional legitimate large transactions
large_legit = rng.random(N_ROWS) < 0.03

amount[large_legit] *= rng.uniform(
    2,
    5,
    large_legit.sum()
)

amount = np.minimum(
    amount,
    balance_before * rng.uniform(
        0.15,
        0.95,
        N_ROWS
    )
)

amount = np.clip(
    amount,
    10,
    150_000
)

# ============================================================
# FEE
# ============================================================

fee_rate = np.where(
    transaction_type == "CASH_OUT",
    0.015,
    0.003
)

fee = amount * fee_rate

amount = np.minimum(
    amount,
    np.maximum(
        balance_before - fee,
        1
    )
)

balance_after = np.maximum(
    balance_before - amount - fee,
    0
)

depletion_ratio = (
    (amount + fee) /
    np.maximum(balance_before, 1)
)

depletion_ratio = np.clip(
    depletion_ratio,
    0,
    1
)

# ============================================================
# NORMAL SECURITY BEHAVIOR
# ============================================================

failed_pin_attempts = rng.choice(
    [0, 1, 2, 3, 4],
    size=N_ROWS,
    p=[0.84, 0.10, 0.035, 0.02, 0.005]
)

otp_resend_count = rng.choice(
    [0, 1, 2, 3, 4],
    size=N_ROWS,
    p=[0.86, 0.09, 0.03, 0.015, 0.005]
)

pin_reset_recently = rng.binomial(
    1,
    0.025,
    N_ROWS
)

device_is_new = rng.binomial(
    1,
    0.07,
    N_ROWS
)

rooted_device = rng.binomial(
    1,
    0.025,
    N_ROWS
)

emulator_detected = rng.binomial(
    1,
    0.008,
    N_ROWS
)

vpn_active = rng.binomial(
    1,
    0.06,
    N_ROWS
)

screen_share_detected = rng.binomial(
    1,
    0.012,
    N_ROWS
)

channel_changed_recently = rng.binomial(
    1,
    0.06,
    N_ROWS
)

# ============================================================
# TRANSACTION VELOCITY
# ============================================================

tx_count_5m = rng.poisson(
    0.7,
    N_ROWS
)

tx_count_15m = (
    tx_count_5m +
    rng.poisson(1.0, N_ROWS)
)

tx_count_1h = (
    tx_count_15m +
    rng.poisson(2.0, N_ROWS)
)

tx_count_24h = (
    tx_count_1h +
    rng.poisson(5.0, N_ROWS)
)

# Legitimate power users
power_users = rng.random(N_ROWS) < 0.025

tx_count_5m[power_users] += rng.integers(
    2,
    6,
    power_users.sum()
)

tx_count_15m[power_users] += rng.integers(
    3,
    8,
    power_users.sum()
)

# ============================================================
# BALANCE INQUIRIES
# ============================================================

balance_inquiry_count_5m = rng.poisson(
    0.5,
    N_ROWS
)

# ============================================================
# TURNAROUND LATENCY
# ============================================================

turnaround_latency_seconds = rng.lognormal(
    mean=np.log(3600),
    sigma=1.25,
    size=N_ROWS
)

turnaround_latency_seconds = np.clip(
    turnaround_latency_seconds,
    10,
    86400
).astype(int)

# ============================================================
# LOCATION
# ============================================================

latitude = rng.uniform(
    20.8,
    26.5,
    N_ROWS
)

longitude = rng.uniform(
    88.0,
    92.7,
    N_ROWS
)

impossible_travel_speed = rng.gamma(
    shape=2,
    scale=20,
    size=N_ROWS
)

# Rare false-positive impossible travel
travel_noise = rng.random(N_ROWS) < 0.003

impossible_travel_speed[travel_noise] = rng.uniform(
    500,
    900,
    travel_noise.sum()
)

# ============================================================
# RECIPIENT RISK
# ============================================================

recipient_risk_score = (
    rng.beta(
        1.5,
        6,
        N_ROWS
    ) * 100
)

# Some legitimate users interact with risky recipients
risky_recipient_noise = rng.random(N_ROWS) < 0.025

recipient_risk_score[risky_recipient_noise] = rng.uniform(
    65,
    95,
    risky_recipient_noise.sum()
)

# ============================================================
# CREATE LATENT FRAUD PROPENSITY
# ============================================================

risk_signal = np.zeros(N_ROWS)

risk_signal += (
    depletion_ratio > 0.90
) * 1.1

risk_signal += (
    failed_pin_attempts >= 3
) * 1.0

risk_signal += (
    otp_resend_count >= 3
) * 0.8

risk_signal += (
    pin_reset_recently == 1
) * 0.8

risk_signal += (
    device_is_new == 1
) * 0.6

risk_signal += (
    rooted_device == 1
) * 0.5

risk_signal += (
    emulator_detected == 1
) * 0.9

risk_signal += (
    vpn_active == 1
) * 0.35

risk_signal += (
    channel_changed_recently == 1
) * 0.6

risk_signal += (
    tx_count_5m >= 5
) * 0.8

risk_signal += (
    turnaround_latency_seconds < 120
) * 0.8

risk_signal += (
    recipient_risk_score >= 70
) * 0.9

risk_signal += (
    impossible_travel_speed >= 500
) * 1.4

risk_signal += (
    is_night == 1
) * 0.15

# Random noise makes classification non-perfect
risk_signal += rng.normal(
    0,
    1.15,
    N_ROWS
)

# ============================================================
# INITIAL FRAUD SELECTION
# ============================================================

# Fraud probability from latent signal
prob = 1 / (
    1 + np.exp(
        -(risk_signal - 4.0)
    )
)

# Rescale approximately toward requested fraud rate
prob = prob * (
    TARGET_FRAUD_RATE /
    max(prob.mean(), 0.0001)
)

prob = np.clip(
    prob,
    0.001,
    0.75
)

fraud_label = rng.binomial(
    1,
    prob
)

# ============================================================
# CREATE DIFFERENT FRAUD TYPES
# ============================================================

fraud_indices = np.where(
    fraud_label == 1
)[0]

fraud_type = np.full(
    N_ROWS,
    "LEGITIMATE",
    dtype=object
)

fraud_types = rng.choice(
    [
        "ACCOUNT_TAKEOVER",
        "MULE_ACTIVITY",
        "DEVICE_FRAUD",
        "VELOCITY_FRAUD",
        "SOCIAL_ENGINEERING",
        "SUBTLE_FRAUD"
    ],
    size=len(fraud_indices),
    p=[
        0.22,
        0.20,
        0.15,
        0.15,
        0.13,
        0.15
    ]
)

fraud_type[fraud_indices] = fraud_types

# ============================================================
# INJECT FRAUD PATTERNS
# ============================================================

# ------------------------------------------------------------
# Account takeover
# ------------------------------------------------------------

idx = np.where(
    fraud_type == "ACCOUNT_TAKEOVER"
)[0]

device_is_new[idx] = rng.binomial(
    1,
    0.72,
    len(idx)
)

pin_reset_recently[idx] = rng.binomial(
    1,
    0.48,
    len(idx)
)

failed_pin_attempts[idx] = rng.integers(
    1,
    6,
    len(idx)
)

vpn_active[idx] = rng.binomial(
    1,
    0.42,
    len(idx)
)

# Not every takeover has impossible travel
travel_attack = rng.random(
    len(idx)
) < 0.35

impossible_travel_speed[
    idx[travel_attack]
] = rng.uniform(
    500,
    1400,
    travel_attack.sum()
)

# ------------------------------------------------------------
# Mule activity
# ------------------------------------------------------------

idx = np.where(
    fraud_type == "MULE_ACTIVITY"
)[0]

recipient_risk_score[idx] = rng.uniform(
    60,
    100,
    len(idx)
)

turnaround_latency_seconds[idx] = rng.integers(
    20,
    900,
    len(idx)
)

fast = rng.random(
    len(idx)
) < 0.6

turnaround_latency_seconds[
    idx[fast]
] = rng.integers(
    10,
    120,
    fast.sum()
)

# ------------------------------------------------------------
# Device fraud
# ------------------------------------------------------------

idx = np.where(
    fraud_type == "DEVICE_FRAUD"
)[0]

device_is_new[idx] = rng.binomial(
    1,
    0.65,
    len(idx)
)

rooted_device[idx] = rng.binomial(
    1,
    0.48,
    len(idx)
)

emulator_detected[idx] = rng.binomial(
    1,
    0.35,
    len(idx)
)

vpn_active[idx] = rng.binomial(
    1,
    0.50,
    len(idx)
)

# ------------------------------------------------------------
# Velocity fraud
# ------------------------------------------------------------

idx = np.where(
    fraud_type == "VELOCITY_FRAUD"
)[0]

tx_count_5m[idx] += rng.integers(
    3,
    9,
    len(idx)
)

tx_count_15m[idx] += rng.integers(
    4,
    12,
    len(idx)
)

balance_inquiry_count_5m[idx] += rng.integers(
    1,
    6,
    len(idx)
)

# ------------------------------------------------------------
# Social engineering
# ------------------------------------------------------------

idx = np.where(
    fraud_type == "SOCIAL_ENGINEERING"
)[0]

otp_resend_count[idx] = rng.integers(
    1,
    6,
    len(idx)
)

screen_share_detected[idx] = rng.binomial(
    1,
    0.48,
    len(idx)
)

channel_changed_recently[idx] = rng.binomial(
    1,
    0.45,
    len(idx)
)

# ------------------------------------------------------------
# Subtle fraud
# Deliberately don't make these obviously fraudulent
# ------------------------------------------------------------

idx = np.where(
    fraud_type == "SUBTLE_FRAUD"
)[0]

recipient_risk_score[idx] += rng.uniform(
    10,
    30,
    len(idx)
)

recipient_risk_score[idx] = np.clip(
    recipient_risk_score[idx],
    0,
    100
)

amount[idx] *= rng.uniform(
    1.1,
    1.8,
    len(idx)
)

# ============================================================
# SOME FRAUD TRANSACTIONS SHOULD LOOK NORMAL
# ============================================================

fraud_indices = np.where(
    fraud_label == 1
)[0]

hard_fraud_count = int(
    len(fraud_indices) * 0.12
)

hard_fraud = rng.choice(
    fraud_indices,
    size=hard_fraud_count,
    replace=False
)

# Reset several obvious features
failed_pin_attempts[hard_fraud] = rng.choice(
    [0, 1, 2],
    hard_fraud_count,
    p=[0.65, 0.25, 0.10]
)

otp_resend_count[hard_fraud] = rng.choice(
    [0, 1, 2],
    hard_fraud_count,
    p=[0.7, 0.2, 0.1]
)

vpn_active[hard_fraud] = rng.binomial(
    1,
    0.15,
    hard_fraud_count
)

device_is_new[hard_fraud] = rng.binomial(
    1,
    0.25,
    hard_fraud_count
)

# ============================================================
# ADD LEGITIMATE ANOMALIES
# ============================================================

legit_indices = np.where(
    fraud_label == 0
)[0]

false_positive_count = int(
    len(legit_indices) * 0.04
)

fp = rng.choice(
    legit_indices,
    size=false_positive_count,
    replace=False
)

# Legitimate but suspicious-looking transactions
device_is_new[fp] = rng.binomial(
    1,
    0.45,
    false_positive_count
)

vpn_active[fp] = rng.binomial(
    1,
    0.30,
    false_positive_count
)

failed_pin_attempts[fp] += rng.binomial(
    2,
    0.30,
    false_positive_count
)

# ============================================================
# RECALCULATE BALANCE
# ============================================================

amount = np.minimum(
    amount,
    np.maximum(
        balance_before - fee,
        1
    )
)

balance_after = np.maximum(
    balance_before - amount - fee,
    0
)

depletion_ratio = np.clip(
    (amount + fee) /
    np.maximum(balance_before, 1),
    0,
    1
)

# ============================================================
# DERIVED AMOUNT FEATURES
# ============================================================

amount_sum_5m = (
    amount
    * np.maximum(tx_count_5m, 1)
    * rng.uniform(
        0.5,
        1.3,
        N_ROWS
    )
)

amount_sum_1h = (
    amount_sum_5m
    + amount
    * rng.uniform(
        0,
        3,
        N_ROWS
    )
)

# ============================================================
# IMPOSSIBLE TRAVEL FLAG
# ============================================================

impossible_travel = (
    impossible_travel_speed >= 500
).astype(int)

# ============================================================
# DEVICE RISK SCORE
# ============================================================

device_risk_score = (
    device_is_new * 14
    + rooted_device * 16
    + emulator_detected * 20
    + vpn_active * 7
    + screen_share_detected * 10
).astype(float)

device_risk_score += rng.normal(
    5,
    5,
    N_ROWS
)

device_risk_score = np.clip(
    device_risk_score,
    0,
    100
)

# ============================================================
# USER BEHAVIOR SCORE
# ============================================================

user_behavior_score = (
    failed_pin_attempts * 4
    + otp_resend_count * 3
    + pin_reset_recently * 10
    + channel_changed_recently * 8
    + np.minimum(
        tx_count_5m * 3,
        20
    )
).astype(float)

user_behavior_score += rng.normal(
    4,
    5,
    N_ROWS
)

user_behavior_score = np.clip(
    user_behavior_score,
    0,
    100
)

# ============================================================
# BLACKLISTS
# ============================================================

blacklisted_device = np.zeros(
    N_ROWS,
    dtype=int
)

blacklisted_agent = np.zeros(
    N_ROWS,
    dtype=int
)

# Very rare even among fraud transactions
fraud_idx = np.where(
    fraud_label == 1
)[0]

blacklisted_device[
    fraud_idx
] = rng.binomial(
    1,
    0.025,
    len(fraud_idx)
)

blacklisted_agent[
    fraud_idx
] = rng.binomial(
    1,
    0.015,
    len(fraud_idx)
)

# Tiny false positive possibility
blacklisted_device[
    legit_indices
] = rng.binomial(
    1,
    0.0004,
    len(legit_indices)
)

# ============================================================
# RULE SCORE
# IMPORTANT: DO NOT USE THIS AS ML INPUT
# ============================================================

rule_risk_score = np.zeros(
    N_ROWS,
    dtype=float
)

rule_risk_score += np.where(
    depletion_ratio >= 0.90,
    20,
    0
)

rule_risk_score += np.where(
    depletion_ratio >= 0.97,
    15,
    0
)

rule_risk_score += np.where(
    failed_pin_attempts >= 3,
    15,
    0
)

rule_risk_score += np.where(
    otp_resend_count >= 3,
    10,
    0
)

rule_risk_score += (
    pin_reset_recently * 15
)

rule_risk_score += (
    device_is_new * 10
)

rule_risk_score += np.where(
    (
        (device_is_new == 1)
        & (amount >= 20_000)
    ),
    15,
    0
)

rule_risk_score += (
    rooted_device * 10
)

rule_risk_score += (
    emulator_detected * 15
)

rule_risk_score += (
    vpn_active * 5
)

rule_risk_score += np.where(
    tx_count_5m > 5,
    15,
    0
)

rule_risk_score += np.where(
    turnaround_latency_seconds < 120,
    15,
    0
)

rule_risk_score += (
    channel_changed_recently * 10
)

rule_risk_score += np.where(
    balance_inquiry_count_5m >= 4,
    10,
    0
)

rule_risk_score += np.where(
    impossible_travel == 1,
    30,
    0
)

rule_risk_score += np.where(
    recipient_risk_score >= 70,
    20,
    0
)

rule_risk_score = np.where(
    (
        (blacklisted_device == 1)
        | (blacklisted_agent == 1)
    ),
    100,
    rule_risk_score
)

rule_risk_score = np.clip(
    rule_risk_score,
    0,
    100
)

# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame({

    "transaction_id": transaction_id,
    "user_id": user_id,
    "receiver_id": receiver_id,
    "device_id": device_id,

    "transaction_type": transaction_type,
    "channel": channel,

    "amount": np.round(amount, 2),
    "fee": np.round(fee, 2),

    "balance_before": np.round(
        balance_before,
        2
    ),

    "balance_after": np.round(
        balance_after,
        2
    ),

    "depletion_ratio": np.round(
        depletion_ratio,
        4
    ),

    "failed_pin_attempts":
        failed_pin_attempts,

    "otp_resend_count":
        otp_resend_count,

    "pin_reset_recently":
        pin_reset_recently,

    "device_is_new":
        device_is_new,

    "rooted_device":
        rooted_device,

    "emulator_detected":
        emulator_detected,

    "vpn_active":
        vpn_active,

    "screen_share_detected":
        screen_share_detected,

    "channel_changed_recently":
        channel_changed_recently,

    "tx_count_5m":
        tx_count_5m,

    "tx_count_15m":
        tx_count_15m,

    "tx_count_1h":
        tx_count_1h,

    "tx_count_24h":
        tx_count_24h,

    "amount_sum_5m":
        np.round(amount_sum_5m, 2),

    "amount_sum_1h":
        np.round(amount_sum_1h, 2),

    "balance_inquiry_count_5m":
        balance_inquiry_count_5m,

    "turnaround_latency_seconds":
        turnaround_latency_seconds,

    "latitude":
        np.round(latitude, 6),

    "longitude":
        np.round(longitude, 6),

    "impossible_travel":
        impossible_travel,

    "impossible_travel_speed":
        np.round(
            impossible_travel_speed,
            2
        ),

    "recipient_risk_score":
        np.round(
            recipient_risk_score,
            2
        ),

    "device_risk_score":
        np.round(
            device_risk_score,
            2
        ),

    "user_behavior_score":
        np.round(
            user_behavior_score,
            2
        ),

    "blacklisted_device":
        blacklisted_device,

    "blacklisted_agent":
        blacklisted_agent,

    "hour":
        hour,

    "day_of_week":
        day_of_week,

    "is_night":
        is_night,

    # Rule engine output
    # Exclude this from ML training
    "rule_risk_score":
        rule_risk_score.astype(int),

    # Helpful for analysis/demo
    # Don't necessarily give this to the model
    "fraud_type":
        fraud_type,

    # ML target
    "fraud_label":
        fraud_label,

    "timestamp":
        timestamp
})

# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)

# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE = "mfs_synthetic_100k_v2.csv"

df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ============================================================
# QUALITY CHECK
# ============================================================

print("=" * 65)
print("MFS SYNTHETIC FRAUD DATASET V2")
print("=" * 65)

print(
    f"\nTransactions     : {len(df):,}"
)

fraud_count = int(
    df["fraud_label"].sum()
)

legit_count = (
    len(df) - fraud_count
)

print(
    f"Legitimate       : {legit_count:,}"
)

print(
    f"Fraudulent       : {fraud_count:,}"
)

print(
    f"Fraud Rate       : "
    f"{df['fraud_label'].mean() * 100:.2f}%"
)

print(
    f"Features          : {len(df.columns)}"
)

print("\nFraud types:")

print(
    df[
        df["fraud_label"] == 1
    ]["fraud_type"]
    .value_counts()
)

print("\nAverage rule risk score:")

print(
    df.groupby(
        "fraud_label"
    )["rule_risk_score"]
    .agg([
        "mean",
        "median",
        "std",
        "min",
        "max"
    ])
)

print("\nRule score overlap:")

print(
    "Legitimate score >= 40:",
    (
        (
            (df["fraud_label"] == 0)
            & (df["rule_risk_score"] >= 40)
        ).sum()
    )
)

print(
    "Fraud score < 40:",
    (
        (
            (df["fraud_label"] == 1)
            & (df["rule_risk_score"] < 40)
        ).sum()
    )
)

print(
    "\nSaved:",
    OUTPUT_FILE
)

print("\nDone.")