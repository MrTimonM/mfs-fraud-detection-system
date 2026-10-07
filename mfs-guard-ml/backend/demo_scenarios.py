"""Four deterministic demo scenarios: history to ingest + the transaction to score.

Inputs are realistic synthetic payloads; thresholds are never adjusted to force outcomes.
Usage (from mfs-guard-ml/): .venv/Scripts/python -m backend.demo_scenarios
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

BASE = datetime(2026, 3, 2, 0, 0, tzinfo=timezone.utc)

PROFILE = {'typical_transaction_amount': 2000.0, 'home_latitude': 23.81, 'home_longitude': 90.41,
           'preferred_channel': 'APP', 'preferred_transaction_type': 'SEND_MONEY',
           'customer_segment': 'SALARIED', 'account_age_days': 900,
           'normal_active_hour_start': 8, 'normal_active_hour_end': 22,
           'normal_location_radius_km': 15.0, 'wallet_balance_tendency': 15.0}

CLEAN = dict(failed_pin_attempts=0, otp_resend_count=0, balance_inquiry_count_5m=0,
             pin_reset_recently=0, password_reset_recently=0, biometric_failed_recently=0,
             rooted_device=0, emulator_detected=0, vpn_active=0, screen_share_detected=0,
             sim_changed_recently=0, device_os_changed=0, device_fingerprint_changed=0,
             blacklisted_device=0, blacklisted_agent=0, fee=0.0)


def _tx(tid, user, receiver, device, ts, amount, balance, ttype='SEND_MONEY', channel='APP',
        profile=PROFILE, lat=23.81, lon=90.41, **kw):
    tx = {'transaction_id': tid, 'user_id': user, 'receiver_id': receiver, 'device_id': device,
          'timestamp': ts.isoformat(), 'transaction_type': ttype, 'channel': channel,
          'amount': float(amount), 'balance_before': float(balance), 'latitude': lat, 'longitude': lon,
          'customer_profile': profile, **CLEAN}
    tx.update(kw)
    return tx


def _regular_history(prefix, user, device, days=12, amount=1800.0):
    """Two ordinary daytime transfers per day to two known recipients from the usual device."""
    hist = []
    for day in range(days):
        for k, (hour, rcv) in enumerate([(10, f'{prefix}_FRIEND'), (18, f'{prefix}_SHOP')]):
            amt = amount * (0.85 + 0.1 * ((day + k) % 4))
            hist.append(_tx(f'{prefix}_H{day:02d}{k}', user, rcv, device,
                            BASE + timedelta(days=day, hours=hour), amt, 140000 - 300 * day))
    return hist


def normal():
    hist = _regular_history('N', 'N_USER', 'N_DEV')
    tx = _tx('N_TX', 'N_USER', 'N_FRIEND', 'N_DEV', BASE + timedelta(days=12, hours=11), 1900, 136000)
    return hist, tx


def moderate():
    hist = _regular_history('M', 'M_USER', 'M_DEV')
    # New recipient, ~4x the customer's usual amount, a couple of OTP resends; otherwise
    # usual device, usual hour, usual location.
    tx = _tx('M_TX', 'M_USER', 'M_NEW_RECIPIENT', 'M_DEV', BASE + timedelta(days=12, hours=20),
             8000, 136000, otp_resend_count=2, balance_inquiry_count_5m=1)
    return hist, tx


def account_takeover():
    hist = _regular_history('A', 'A_USER', 'A_DEV')
    # Attacker on a new device at 03:00 local-UTC, after failed PINs and a PIN reset,
    # draining most of the balance to a never-seen recipient from an unusual location.
    tx = _tx('A_TX', 'A_USER', 'A_UNKNOWN_RECIPIENT', 'A_ATTACKER_DEV', BASE + timedelta(days=12, hours=3),
             30000, 33000, failed_pin_attempts=4, otp_resend_count=3, pin_reset_recently=1,
             password_reset_recently=1, vpn_active=1, device_fingerprint_changed=1,
             sim_changed_recently=1, balance_inquiry_count_5m=3, lat=22.35, lon=91.78)
    return hist, tx


MULE_PROFILE = {**PROFILE, 'customer_segment': 'STUDENT', 'account_age_days': 25,
                'typical_transaction_amount': 1500.0}


def mule():
    # Recently opened student wallet with a short, ordinary history.
    hist = [{**h, 'customer_profile': MULE_PROFILE} for h in
            _regular_history('X', 'X_MULE', 'X_MULE_DEV', days=6, amount=1500.0)]
    t0 = BASE + timedelta(days=6, hours=13)
    # 15 unrelated senders push funds into the mule wallet within ~30 minutes.
    for i in range(15):
        hist.append(_tx(f'X_IN{i:02d}', f'X_SENDER{i:02d}', 'X_MULE', f'X_SDEV{i:02d}',
                        t0 + timedelta(minutes=2 * i), 4000 + 150 * i, 60000))
    # A collector wallet upstream already receives from many mule wallets.
    for i in range(16):
        hist.append(_tx(f'X_HUBIN{i:02d}', f'X_OTHERMULE{i:02d}', 'X_HUB', f'X_ODEV{i:02d}',
                        t0 + timedelta(minutes=2 * i + 1), 9000, 20000))
    # The mule then rapidly forwards: three quick forwards already happened...
    # Forwards hop between channels (APP/USSD) -- a common evasion pattern.
    for j in range(3):
        hist.append(_tx(f'X_FW{j}', 'X_MULE', 'X_HUB', 'X_MULE_DEV',
                        t0 + timedelta(minutes=31 + j), 18000, 90000 - 18000 * j,
                        channel='USSD' if j % 2 == 0 else 'APP', profile=MULE_PROFILE))
    # ...and the scored transaction is the next forward to the collector, draining the rest.
    tx = _tx('X_TX', 'X_MULE', 'X_HUB', 'X_MULE_DEV', t0 + timedelta(minutes=34),
             34500, 36000, channel='APP', profile=MULE_PROFILE, balance_inquiry_count_5m=3)
    return hist, tx


SCENARIOS = {'normal': normal, 'moderate': moderate, 'account_takeover': account_takeover, 'mule': mule}
EXPECTED = {'normal': {'APPROVE'}, 'moderate': {'STEP_UP_AUTH'},
            'account_takeover': {'TEMPORARY_HOLD', 'REJECT_AND_FREEZE'},
            'mule': {'TEMPORARY_HOLD', 'REJECT_AND_FREEZE'}}


def run(client):
    results = {}
    for name, fn in SCENARIOS.items():
        hist, tx = fn()
        client.post('/api/v1/history/ingest', json={'transactions': hist}).raise_for_status()
        r = client.post('/api/v1/transactions/analyze', json=tx)
        r.raise_for_status()
        results[name] = r.json()
    return results


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    from fastapi.testclient import TestClient
    from backend.app.main import create_app
    out = run(TestClient(create_app()))
    for name, r in out.items():
        print(f"{name:17s} p={r['fraud_probability']} level={r['risk_level']} rule={r['rule_risk_score']} "
              f"decision={r['decision']} rules={r['triggered_rules']}")
        print('   factors:', r['top_risk_factors'])
