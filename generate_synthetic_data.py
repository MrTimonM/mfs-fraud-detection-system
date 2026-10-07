import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
N_ROWS = 100_000
FRAUD_RATE = 0.07

np.random.seed(SEED)

# ============================================================
# BASIC IDENTIFIERS
# ============================================================

transaction_ids = [f"TXN{i:07d}" for i in range(1, N_ROWS + 1)]

user_ids = np.array([
    f"USR{num:06d}"
    for num in np.random.randint(1, 25_001, N_ROWS)
])

receiver_ids = np.array([
    f"RCV{num:06d}"
    for num in np.random.randint(1, 15_001, N_ROWS)
])

device_ids = np.array([
    f"DEV{num:06d}"
    for num in np.random.randint(1, 30_001, N_ROWS)
])

# ============================================================
# FRAUD LABEL
# 0 = Legitimate
# 1 = Fraud
# ============================================================

fraud_label = np.random.choice(
    [0, 1],
    size=N_ROWS,
    p=[1 - FRAUD_RATE, FRAUD_RATE]
)

fraud_mask = fraud_label == 1
normal_mask = fraud_label == 0

# ============================================================
# TRANSACTION TYPE
# ============================================================

transaction_types = np.random.choice(
    ["SEND_MONEY", "CASH_OUT", "PAYMENT", "MOBILE_RECHARGE"],
    size=N_ROWS,
    p=[0.35, 0.30, 0.25, 0.10]
)

# Fraud transactions are more likely to be cash-out/send-money
transaction_types[fraud_mask] = np.random.choice(
    ["SEND_MONEY", "CASH_OUT"],
    size=fraud_mask.sum(),
    p=[0.35, 0.65]
)

# ============================================================
# CHANNEL
# ============================================================

channels = np.random.choice(
    ["APP", "USSD", "AGENT"],
    size=N_ROWS,
    p=[0.60, 0.20, 0.20]
)

# ============================================================
# ACCOUNT BALANCE
# ============================================================

balance_before = np.random.lognormal(
    mean=9.0,
    sigma=0.8,
    size=N_ROWS
)

balance_before = np.clip(
    balance_before,
    500,
    200_000
).round(2)

# ============================================================
# TRANSACTION AMOUNT
# ============================================================

amount = np.zeros(N_ROWS)

# Normal transactions
normal_ratio = np.random.beta(
    1.5,
    6,
    normal_mask.sum()
)

amount[normal_mask] = (
    balance_before[normal_mask] * normal_ratio
)

# Fraud transactions often try to empty account
fraud_ratio = np.random.uniform(
    0.65,
    0.995,
    fraud_mask.sum()
)

amount[fraud_mask] = (
    balance_before[fraud_mask] * fraud_ratio
)

amount = np.maximum(amount, 10)
amount = np.round(amount, 2)

# ============================================================
# TRANSACTION FEE
# ============================================================

fee = np.where(
    transaction_types == "CASH_OUT",
    amount * 0.015,
    amount * 0.003
)

fee = np.round(fee, 2)

# Prevent amount + fee from exceeding balance
amount = np.minimum(
    amount,
    balance_before - fee
)

amount = np.maximum(amount, 1)
amount = np.round(amount, 2)

balance_after = np.maximum(
    balance_before - amount - fee,
    0
).round(2)

# ============================================================
# DEPLETION RATIO
# ============================================================

depletion_ratio = (
    (amount + fee) / balance_before
)

depletion_ratio = np.clip(
    depletion_ratio,
    0,
    1
).round(4)

# ============================================================
# SECURITY FEATURES
# ============================================================

failed_pin_attempts = np.zeros(N_ROWS, dtype=int)

failed_pin_attempts[normal_mask] = np.random.choice(
    [0, 1, 2],
    normal_mask.sum(),
    p=[0.88, 0.10, 0.02]
)

failed_pin_attempts[fraud_mask] = np.random.choice(
    [0, 1, 2, 3, 4, 5],
    fraud_mask.sum(),
    p=[0.05, 0.08, 0.12, 0.25, 0.25, 0.25]
)

# OTP resend
otp_resend_count = np.zeros(N_ROWS, dtype=int)

otp_resend_count[normal_mask] = np.random.choice(
    [0, 1, 2],
    normal_mask.sum(),
    p=[0.90, 0.08, 0.02]
)

otp_resend_count[fraud_mask] = np.random.choice(
    [0, 1, 2, 3, 4, 5],
    fraud_mask.sum(),
    p=[0.05, 0.08, 0.12, 0.25, 0.25, 0.25]
)

# ============================================================
# BOOLEAN RISK FEATURES
# ============================================================

def generate_binary(normal_probability, fraud_probability):
    result = np.zeros(N_ROWS, dtype=int)

    result[normal_mask] = np.random.binomial(
        1,
        normal_probability,
        normal_mask.sum()
    )

    result[fraud_mask] = np.random.binomial(
        1,
        fraud_probability,
        fraud_mask.sum()
    )

    return result


pin_reset_recently = generate_binary(0.01, 0.45)
device_is_new = generate_binary(0.05, 0.62)
rooted_device = generate_binary(0.015, 0.30)
emulator_detected = generate_binary(0.005, 0.20)
vpn_active = generate_binary(0.03, 0.48)
screen_share_detected = generate_binary(0.005, 0.20)
channel_changed_recently = generate_binary(0.04, 0.42)
blacklisted_device = generate_binary(0.001, 0.08)
blacklisted_agent = generate_binary(0.001, 0.05)

# ============================================================
# TRANSACTION VELOCITY
# ============================================================

tx_count_5m = np.zeros(N_ROWS, dtype=int)

tx_count_5m[normal_mask] = np.random.poisson(
    0.5,
    normal_mask.sum()
)

tx_count_5m[fraud_mask] = np.random.poisson(
    5,
    fraud_mask.sum()
)

tx_count_15m = tx_count_5m + np.random.poisson(
    1,
    N_ROWS
)

tx_count_1h = tx_count_15m + np.random.poisson(
    2,
    N_ROWS
)

tx_count_24h = tx_count_1h + np.random.poisson(
    5,
    N_ROWS
)

# ============================================================
# AMOUNT VELOCITY
# ============================================================

amount_sum_5m = (
    amount *
    np.maximum(tx_count_5m, 1) *
    np.random.uniform(0.5, 1.3, N_ROWS)
).round(2)

amount_sum_1h = (
    amount_sum_5m +
    amount *
    np.random.uniform(0, 3, N_ROWS)
).round(2)

# ============================================================
# BALANCE INQUIRY COUNT
# ============================================================

balance_inquiry_count_5m = np.zeros(
    N_ROWS,
    dtype=int
)

balance_inquiry_count_5m[normal_mask] = np.random.poisson(
    0.4,
    normal_mask.sum()
)

balance_inquiry_count_5m[fraud_mask] = np.random.poisson(
    4,
    fraud_mask.sum()
)

# ============================================================
# TURNAROUND LATENCY
# Time between receiving money and sending/cashing it out
# ============================================================

turnaround_latency_seconds = np.zeros(
    N_ROWS,
    dtype=int
)

turnaround_latency_seconds[normal_mask] = np.random.randint(
    300,
    86400,
    normal_mask.sum()
)

turnaround_latency_seconds[fraud_mask] = np.random.randint(
    10,
    900,
    fraud_mask.sum()
)

# Some fraud cases specifically rapid turnaround
rapid_indices = np.where(fraud_mask)[0]

selected_rapid = np.random.choice(
    rapid_indices,
    size=int(len(rapid_indices) * 0.60),
    replace=False
)

turnaround_latency_seconds[selected_rapid] = np.random.randint(
    10,
    120,
    len(selected_rapid)
)

# ============================================================
# LOCATION / IMPOSSIBLE TRAVEL
# ============================================================

impossible_travel_speed = np.zeros(N_ROWS)

impossible_travel_speed[normal_mask] = np.random.uniform(
    0,
    100,
    normal_mask.sum()
)

impossible_travel_speed[fraud_mask] = np.random.uniform(
    50,
    1200,
    fraud_mask.sum()
)

impossible_travel = np.zeros(N_ROWS, dtype=int)

# normal users rarely trigger
impossible_travel[normal_mask] = np.random.binomial(
    1,
    0.002,
    normal_mask.sum()
)

# many fraud cases trigger it
impossible_travel[fraud_mask] = np.random.binomial(
    1,
    0.38,
    fraud_mask.sum()
)

# Make impossible travel speed clearly abnormal
impossible_indices = impossible_travel == 1

impossible_travel_speed[impossible_indices] = np.random.uniform(
    500,
    1500,
    impossible_indices.sum()
)

impossible_travel_speed = np.round(
    impossible_travel_speed,
    2
)

# ============================================================
# RECIPIENT RISK
# ============================================================

recipient_risk_score = np.zeros(N_ROWS)

recipient_risk_score[normal_mask] = np.random.beta(
    1.5,
    8,
    normal_mask.sum()
) * 100

recipient_risk_score[fraud_mask] = np.random.beta(
    5,
    2,
    fraud_mask.sum()
) * 100

recipient_risk_score = np.round(
    recipient_risk_score,
    2
)

# ============================================================
# DEVICE RISK
# ============================================================

device_risk_score = (
    device_is_new * 20
    + rooted_device * 20
    + emulator_detected * 25
    + vpn_active * 10
    + screen_share_detected * 15
    + blacklisted_device * 50
).astype(float)

device_risk_score += np.random.uniform(
    0,
    10,
    N_ROWS
)

device_risk_score = np.clip(
    device_risk_score,
    0,
    100
).round(2)

# ============================================================
# USER BEHAVIOR SCORE
# ============================================================

user_behavior_score = (
    failed_pin_attempts * 5
    + otp_resend_count * 4
    + pin_reset_recently * 15
    + channel_changed_recently * 12
    + np.minimum(tx_count_5m * 4, 25)
).astype(float)

user_behavior_score += np.random.uniform(
    0,
    8,
    N_ROWS
)

user_behavior_score = np.clip(
    user_behavior_score,
    0,
    100
).round(2)

# ============================================================
# RULE RISK SCORE
# Similar to project's rule engine
# ============================================================

rule_risk_score = np.zeros(N_ROWS)

rule_risk_score += np.where(
    depletion_ratio >= 0.90,
    20,
    0
)

rule_risk_score += np.where(
    depletion_ratio >= 0.97,
    35,
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

rule_risk_score += pin_reset_recently * 15

rule_risk_score += device_is_new * 10

rule_risk_score += (
    (device_is_new == 1) &
    (amount >= 20_000)
) * 25

rule_risk_score += rooted_device * 15
rule_risk_score += emulator_detected * 20
rule_risk_score += vpn_active * 8

rule_risk_score += np.where(
    tx_count_5m > 5,
    20,
    0
)

rule_risk_score += np.where(
    turnaround_latency_seconds < 120,
    25,
    0
)

rule_risk_score += channel_changed_recently * 15

rule_risk_score += np.where(
    balance_inquiry_count_5m >= 4,
    15,
    0
)

rule_risk_score += impossible_travel * 40

rule_risk_score += np.where(
    recipient_risk_score >= 70,
    25,
    0
)

# Immediate rejection conditions
rule_risk_score = np.where(
    (blacklisted_device == 1) |
    (blacklisted_agent == 1),
    100,
    rule_risk_score
)

rule_risk_score = np.clip(
    rule_risk_score,
    0,
    100
).astype(int)

# ============================================================
# TIMESTAMP
# Generate data from last 90 days
# ============================================================

start_date = datetime.now() - timedelta(days=90)

random_seconds = np.random.randint(
    0,
    90 * 24 * 60 * 60,
    N_ROWS
)

timestamps = [
    start_date + timedelta(seconds=int(x))
    for x in random_seconds
]

# ============================================================
# LATITUDE / LONGITUDE
# Around Bangladesh for demo purposes
# ============================================================

latitude = np.random.uniform(
    20.8,
    26.5,
    N_ROWS
).round(6)

longitude = np.random.uniform(
    88.0,
    92.7,
    N_ROWS
).round(6)

# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame({

    # IDs
    "transaction_id": transaction_ids,
    "user_id": user_ids,
    "receiver_id": receiver_ids,
    "device_id": device_ids,

    # Transaction
    "transaction_type": transaction_types,
    "channel": channels,
    "amount": amount,
    "fee": fee,
    "balance_before": balance_before,
    "balance_after": balance_after,
    "depletion_ratio": depletion_ratio,

    # Authentication / security
    "failed_pin_attempts": failed_pin_attempts,
    "otp_resend_count": otp_resend_count,
    "pin_reset_recently": pin_reset_recently,

    # Device
    "device_is_new": device_is_new,
    "rooted_device": rooted_device,
    "emulator_detected": emulator_detected,
    "vpn_active": vpn_active,
    "screen_share_detected": screen_share_detected,

    # Channel behavior
    "channel_changed_recently": channel_changed_recently,

    # Velocity
    "tx_count_5m": tx_count_5m,
    "tx_count_15m": tx_count_15m,
    "tx_count_1h": tx_count_1h,
    "tx_count_24h": tx_count_24h,

    # Amount velocity
    "amount_sum_5m": amount_sum_5m,
    "amount_sum_1h": amount_sum_1h,

    # Behavioral features
    "balance_inquiry_count_5m": balance_inquiry_count_5m,
    "turnaround_latency_seconds": turnaround_latency_seconds,

    # Location
    "latitude": latitude,
    "longitude": longitude,
    "impossible_travel": impossible_travel,
    "impossible_travel_speed": impossible_travel_speed,

    # Recipient / account risk
    "recipient_risk_score": recipient_risk_score,
    "device_risk_score": device_risk_score,
    "user_behavior_score": user_behavior_score,

    # Blacklists
    "blacklisted_device": blacklisted_device,
    "blacklisted_agent": blacklisted_agent,

    # Existing rule engine result
    "rule_risk_score": rule_risk_score,

    # Metadata
    "timestamp": timestamps,

    # ML target
    "fraud_label": fraud_label
})

# ============================================================
# SORT BY TIME
# ============================================================

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)

# ============================================================
# SAVE DATASET
# ============================================================

OUTPUT_FILE = "mfs_synthetic_100k.csv"

df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

print("=" * 60)
print("Synthetic MFS Fraud Dataset Generated")
print("=" * 60)

print(f"\nTotal transactions : {len(df):,}")

print(
    f"Legitimate         : "
    f"{(df['fraud_label'] == 0).sum():,}"
)

print(
    f"Fraudulent         : "
    f"{(df['fraud_label'] == 1).sum():,}"
)

print(
    f"Fraud rate         : "
    f"{df['fraud_label'].mean() * 100:.2f}%"
)

print(
    f"Number of features : "
    f"{len(df.columns)}"
)

print(
    f"\nSaved as: {OUTPUT_FILE}"
)

print("\nFraud label distribution:")

print(
    df["fraud_label"]
    .value_counts()
    .sort_index()
)

print("\nAverage rule risk score:")

print(
    df.groupby("fraud_label")[
        "rule_risk_score"
    ].mean()
)

print("\nSample rows:")

print(
    df.head()
)

print("\nDone.")