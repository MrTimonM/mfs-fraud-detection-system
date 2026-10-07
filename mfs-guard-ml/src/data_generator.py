"""Persistent-wallet event simulation; labels never feed feature computation."""
import logging
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from .config import config, paths, ROOT, dump
from .entity_generator import entities, TYPES, CHANNELS
from .fraud_scenarios import episodes, FRAUD_TYPES
from .feature_engineering import engineer
from .rule_engine import apply_rules


def generate(rows, seed=42, save=True):
    cfg = config()
    rng = np.random.default_rng(seed)
    n = max(80, rows // cfg['transactions_per_user'])
    profiles = entities(n, rng)
    groups = (rows + 2) // 3
    users = rng.integers(n, size=groups)
    observed, subtype = episodes(groups, rng)
    day = rng.integers(cfg['simulation_days'], size=groups)
    hour = rng.integers(7, 23, size=groups)
    hour = np.where(rng.random(groups) < .08, rng.integers(24, size=groups), hour)
    base = day * 86400 + hour * 3600 + rng.uniform(0, 3600, groups)
    burst = (rng.random(groups) < np.where(observed == 3, .8, .28))
    offset = rng.uniform(3000, 70000, groups)
    offset = np.where(burst, rng.uniform(20, 100, groups), offset)
    seconds = (base[:, None] + offset[:, None] * np.arange(3)).ravel()[:rows]
    uid = np.repeat(users, 3)[:rows]
    scenario = np.repeat(observed, 3)[:rows]
    subtype = np.repeat(subtype, 3)[:rows]
    order = np.argsort(seconds, kind='stable')
    seconds, uid, scenario, subtype = [x[order] for x in (seconds, uid, scenario, subtype)]
    # Make ordering strictly total even in the unlikely event of equal timestamps.
    seconds += np.arange(rows) * 1e-7
    profile = profiles.iloc[uid].reset_index(drop=True)
    signal = rng.random(rows) < .7
    unusual = (scenario > 0) & signal
    amount = profile.typical_transaction_amount.to_numpy() * rng.lognormal(0, .65, rows)
    amount *= np.where(np.isin(scenario, [1, 6, 8, 9]) & signal, rng.uniform(1.5, 3.5, rows), 1)
    amount *= np.where(scenario == 3, .65, 1)
    amount *= 1 + .12 * seconds / (cfg['simulation_days'] * 86400)
    tx = rng.choice(TYPES, rows, p=[.30, .18, .22, .10, .07, .05, .05, .03])
    preferred = profile.preferred_transaction_type.to_numpy()
    choose = rng.random(rows) < .15
    tx[choose] = preferred[choose]
    tx[np.isin(scenario, [7, 8]) & signal] = 'CASH_OUT'
    tx[np.isin(scenario, [1, 2, 6, 9]) & signal] = 'SEND_MONEY'
    channel = profile.preferred_channel.to_numpy().copy()
    changed = rng.random(rows) < np.where(unusual, .35, .13)
    channel[changed] = rng.choice(CHANNELS, changed.sum())
    channel[tx == 'CASH_OUT'] = 'AGENT'
    common = (uid * 7 + rng.integers(1, 6, rows)) % n
    receiver = np.where(rng.random(rows) < np.where(unusual, .62, .17), rng.integers(n, size=rows), common)
    # Concentrated recipient pools overlap merchant hubs; mule wallets also send.
    hubs = max(5, n // 60)
    concentrated = ((scenario == 2) & signal) | (rng.random(rows) < .025)
    receiver[concentrated] = rng.integers(hubs, size=concentrated.sum())
    receiver = np.where(receiver == uid, (receiver + 1) % n, receiver)
    new = rng.random(rows) < np.where(np.isin(scenario, [1, 5, 9]) & signal, .6, .055)
    device = np.where(new, n + rng.integers(max(50, n // 3), size=rows), uid)
    shared = (scenario == 2) & (rng.random(rows) < .35)
    device[shared] = n + (uid[shared] % 20)
    d = pd.DataFrame({'transaction_id': [f'TX{seed}_{i:09d}' for i in range(rows)],
        'user_id': uid, 'receiver_id': receiver, 'device_id': device,
        'session_id': np.repeat(np.arange(groups), 3)[:rows][order],
        'timestamp': pd.Timestamp('2026-01-01', tz='UTC') + pd.to_timedelta(seconds, unit='s'),
        'transaction_type': tx, 'channel': channel, 'currency': 'BDT',
        'amount': np.clip(amount, 10, 100000),
        'agent_id': np.where(channel == 'AGENT', rng.integers(max(20, n // 15), size=rows), -1),
        'merchant_id': np.where(tx == 'MERCHANT_PAYMENT', rng.integers(max(20, n // 10), size=rows), -1),
        'fraud_label': (subtype > 0).astype('int8'), 'fraud_type': FRAUD_TYPES[subtype]})
    for c in profiles.columns.drop('user_id'):
        d[c] = profile[c].to_numpy()
    specs = {'failed_pin_attempts': ([1, 9], .14, 1.8),
             'otp_resend_count': ([1, 6, 9], .12, 1.5),
             'balance_inquiry_count_5m': ([1, 8], .2, 2.0)}
    for c, (scenarios, low, high) in specs.items():
        d[c] = rng.poisson(np.where(np.isin(scenario, scenarios) & signal, high, low))
    flags = {'pin_reset_recently': [1, 9], 'password_reset_recently': [1, 9],
        'biometric_failed_recently': [1], 'rooted_device': [5], 'emulator_detected': [5],
        'vpn_active': [1, 5], 'screen_share_detected': [5, 6], 'sim_changed_recently': [9],
        'device_os_changed': [5], 'device_fingerprint_changed': [1, 5]}
    for c, sc in flags.items():
        d[c] = (rng.random(rows) < np.where(np.isin(scenario, sc) & signal, .45, .035)).astype('int8')
    # Pre-existing reputation lists are sampled independently of latent labels.
    bad_devices = set(rng.choice(np.arange(n, n + max(50, n // 3)), max(2, n // 200), replace=False))
    bad_agents = set(rng.choice(max(20, n // 15), 2, replace=False))
    d['blacklisted_device'] = np.isin(device, list(bad_devices)).astype('int8')
    d['blacklisted_agent'] = d.agent_id.isin(bad_agents).astype('int8')
    travel = rng.random(rows) < np.where(np.isin(scenario, [1, 9]) & signal, .4, .035)
    d['latitude'] = profile.home_latitude + rng.normal(0, .018, rows) + travel * rng.normal(0, 1, rows)
    d['longitude'] = profile.home_longitude + rng.normal(0, .018, rows) + travel * rng.normal(0, 1, rows)
    balances = profiles.typical_transaction_amount.to_numpy() * 70
    before = np.empty(rows)
    after = np.empty(rows)
    values = d.amount.to_numpy().copy()
    fees = np.zeros(rows)
    for i in range(rows):
        u, r = uid[i], receiver[i]
        incoming = tx[i] in ('CASH_IN', 'REMITTANCE')
        before[i] = balances[u]
        rate = .015 if tx[i] == 'CASH_OUT' else .002
        if incoming:
            values[i] *= 2.2
            balances[u] += values[i]
        else:
            values[i] = min(values[i], balances[u] / (1 + rate) * .98)
            fees[i] = values[i] * rate
            balances[u] -= values[i] + fees[i]
            if tx[i] in ('SEND_MONEY', 'BANK_TRANSFER'):
                balances[r] += values[i]
        after[i] = balances[u]
    d['amount'], d['fee'] = values, fees
    d['balance_before'], d['balance_after'] = before, after
    d['depletion_ratio'] = (values + fees) / np.maximum(before, 1e-6)
    d['balance_change_ratio'] = (after - before) / np.maximum(before, 1e-6)
    # Quantize source observations before deriving features, so persisted raw
    # values reproduce the same rolling and geodesic calculations on replay.
    for c in d.select_dtypes('float64'):
        d[c] = d[c].astype('float32')
    started = time.perf_counter()
    d = engineer(d, cfg)
    apply_rules(d, cfg)
    feature_seconds = time.perf_counter() - started
    logging.info('Engineered %d rows in %.1fs', rows, feature_seconds)
    for c in d.select_dtypes('float64'):
        d[c] = d[c].astype('float32')
    for c in d.select_dtypes('int64'):
        d[c] = d[c].astype('int32')
    if save:
        p = paths(rows)
        d.to_parquet(ROOT / f'data/raw/mfs_{p["tag"]}.parquet', index=False)
        d.to_csv(ROOT / f'data/raw/mfs_{p["tag"]}.csv', index=False)
        profiles.to_parquet(ROOT / f'data/metadata/profiles_{p["tag"]}.parquet', index=False)
        dump(ROOT / f'data/metadata/generation_{p["tag"]}.json',
             {'seed': seed, 'rows': rows, 'users': n, 'config': cfg,
              'generated_at_utc': datetime.now(timezone.utc).isoformat(),
              'feature_engineering_seconds': feature_seconds,
              'feature_engineering_scope': 'offline chronological replay, not online state retrieval',
              'method': 'Persistent wallets, latent three-event episodes, overlapping stress, pre-event histories'})
    return d
