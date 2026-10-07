"""Persistent profiles; fewer users intentionally provide useful history."""
import numpy as np
import pandas as pd

TYPES = np.array(['SEND_MONEY', 'CASH_OUT', 'CASH_IN', 'MERCHANT_PAYMENT',
                  'MOBILE_RECHARGE', 'BILL_PAYMENT', 'BANK_TRANSFER', 'REMITTANCE'])
CHANNELS = np.array(['APP', 'USSD', 'AGENT', 'WEB', 'API'])


def entities(n, rng):
    return pd.DataFrame({
        'user_id': np.arange(n),
        'typical_transaction_amount': np.clip(rng.lognormal(7.3, .7, n), 100, 15000),
        'home_latitude': rng.uniform(22.2, 25.5, n),
        'home_longitude': rng.uniform(89, 91.5, n),
        'preferred_channel': rng.choice(CHANNELS, n, p=[.54, .25, .16, .03, .02]),
        'preferred_transaction_type': rng.choice(TYPES, n),
        'customer_segment': rng.choice(['REGULAR', 'MERCHANT', 'STUDENT', 'SALARIED'], n),
        'account_age_days': rng.integers(30, 2500, n),
        'normal_active_hour_start': rng.integers(6, 11, n),
        'normal_active_hour_end': rng.integers(19, 24, n),
        'normal_location_radius_km': rng.uniform(5, 50, n),
        'wallet_balance_tendency': rng.uniform(8, 25, n),
    })
