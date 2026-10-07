"""Streaming past-only histories. No function reads labels or fraud types."""
from bisect import bisect_left
from collections import defaultdict
import numpy as np
import pandas as pd
from .behavioral_features import baseline
from .graph_features import HistoricalGraph
from .sequence_features import sequences

WINDOWS = {'5m': 300, '15m': 900, '1h': 3600, '6h': 21600, '24h': 86400}


def haversine(a, b, c, d):
    a, b, c, d = np.radians([a, b, c, d])
    x = np.sin((c-a)/2)**2 + np.cos(a)*np.cos(c)*np.sin((d-b)/2)**2
    return float(6371 * 2 * np.arcsin(np.sqrt(np.clip(x, 0, 1))))


class History:
    def __init__(self):
        self.t, self.a, self.other = [], [], []
        self.total = [0.]

    def add(self, t, a, other=0):
        self.t.append(t)
        self.a.append(a)
        self.other.append(other)
        self.total.append(self.total[-1] + a)

    def window(self, t, seconds):
        start = bisect_left(self.t, t - seconds)
        return len(self.t) - start, self.total[-1] - self.total[start], start


def engineer(d, cfg):
    user = defaultdict(History)
    receiver = defaultdict(History)
    device = defaultdict(History)
    agent = defaultdict(History)
    relation = defaultdict(History)
    device_users, user_devices = defaultdict(set), defaultdict(set)
    last, incoming, outgoing = {}, defaultdict(History), defaultdict(History)
    graph = HistoricalGraph()
    features = None
    # Explicitly drop outcomes at the boundary to enforce target-blind features.
    observable = d.drop(columns=['fraud_label', 'fraud_type'])
    columns = list(observable.columns)
    times = d.timestamp.astype('int64').to_numpy() / (1e6 if d.timestamp.dtype.unit == 'us' else 1e9)
    for i, vals in enumerate(observable.itertuples(index=False, name=None)):
        row = dict(zip(columns, vals))
        t, u, r, dev, a = times[i], row['user_id'], row['receiver_id'], row['device_id'], row['amount']
        h, rh, dh = user[u], receiver[r], device[dev]
        f = baseline(h.a, a, row['typical_transaction_amount'])
        previous = last.get(u)
        dt = t - h.t[-1] if h.t else 1e9
        f['time_since_previous_tx_seconds'] = dt
        f['behavioral_model_active'] = int(len(h.t) >= cfg['minimum_history'])
        for label, duration in WINDOWS.items():
            count, total, start = h.window(t, duration)
            f[f'tx_count_{label}'], f[f'amount_sum_{label}'] = count, total
            if label in ('1h', '24h'):
                f[f'avg_amount_{label}'] = total / max(count, 1)
        f['max_amount_24h'] = max(h.a[h.window(t, 86400)[2]:], default=0)
        f['user_max_amount_30d'] = max(h.a[h.window(t, 2592000)[2]:], default=0)
        days = max(1, (t - h.t[0]) / 86400) if h.t else 1
        f['user_mean_daily_tx_count'], f['user_mean_hourly_tx_count'] = len(h.t)/days, len(h.t)/(24*days)
        f['device_is_new'] = int(dev not in user_devices[u])
        f['device_age_days'] = (t - dh.t[0])/86400 if dh.t else 0
        f['device_first_seen'] = int(not dh.t)
        f['shared_device_user_count'] = len(device_users[dev])
        f['user_device_count'] = len(user_devices[u])
        f['device_transaction_count_24h'] = dh.window(t, 86400)[0]
        f['device_transaction_count_7d'] = dh.window(t, 604800)[0]
        f['channel_changed_recently'] = int(previous is not None and previous['channel'] != row['channel'] and dt < 86400)
        f['preferred_channel_mismatch'] = int(row['channel'] != row['preferred_channel'])
        rel = relation[(u, r)]
        f['first_time_recipient'] = int(not rel.t)
        f['recipient_frequency_user'] = len(rel.t) / max(len(h.t), 1)
        f['user_receiver_tx_count_7d'] = rel.window(t, 604800)[0]
        f['user_receiver_tx_count_30d'] = rel.window(t, 2592000)[0]
        f['time_since_last_recipient_tx'] = t-rel.t[-1] if rel.t else 1e9
        for suffix, duration in [('24h', 86400), ('7d', 604800)]:
            count, total, start = rh.window(t, duration)
            f[f'receiver_transaction_count_{suffix}'] = count
            f[f'receiver_unique_senders_{suffix}'] = len(set(rh.other[start:]))
        for suffix, duration in [('1h', 3600), ('24h', 86400)]:
            f[f'recent_inflow_amount_{suffix}'] = incoming[u].window(t, duration)[1]
            f[f'recent_outflow_amount_{suffix}'] = outgoing[u].window(t, duration)[1]
            f[f'receiver_inflow_{suffix}'] = incoming[r].window(t, duration)[1]
            f[f'receiver_outflow_{suffix}'] = outgoing[r].window(t, duration)[1]
        turnaround = t-incoming[u].t[-1] if incoming[u].t else 1e9
        f['turnaround_latency_seconds'] = turnaround
        f['rapid_fund_turnaround'] = int(turnaround < 600)
        f['receiver_turnaround_ratio'] = min(10, f['receiver_outflow_1h']/max(f['receiver_inflow_1h'], 1))
        f['receiver_sender_diversity'] = f['receiver_unique_senders_7d']/max(f['receiver_transaction_count_7d'], 1)
        f.update(graph.observe(u, r))
        f['fund_concentration'] = f['receiver_inflow_24h']/max(1, f['receiver_unique_senders_24h'])
        f['rapid_forwarding_ratio'] = f['receiver_turnaround_ratio']
        f['shared_device_count'] = f['shared_device_user_count']
        f['mule_network_score'] = min(100, 4*f['receiver_unique_senders_24h'] + 8*f['receiver_turnaround_ratio'])
        f['recipient_risk_score'] = min(100, 5*f['receiver_unique_senders_24h'] + 5*f['receiver_turnaround_ratio'])
        f['device_risk_score'] = min(100, 15*row['rooted_device'] + 20*row['emulator_detected'] + 4*f['shared_device_user_count'] + 25*row['blacklisted_device'])
        lat, lon = row['latitude'], row['longitude']
        plat, plon = (previous['latitude'], previous['longitude']) if previous else (lat, lon)
        distance = haversine(plat, plon, lat, lon)
        f.update(previous_latitude=plat, previous_longitude=plon,
                 distance_from_previous_tx_km=distance, time_since_previous_location_seconds=dt,
                 distance_from_home_km=haversine(row['home_latitude'], row['home_longitude'], lat, lon),
                 impossible_travel_speed=distance/max(dt/3600, 1e-6))
        f['impossible_travel'] = int(f['impossible_travel_speed'] > 900 and distance > 20)
        hour = row['timestamp'].hour
        f['transaction_hour'], f['day_of_week'] = hour, row['timestamp'].dayofweek
        f['weekend_flag'] = int(row['timestamp'].dayofweek in (4, 5))
        f['is_night_transaction'] = int(hour < 6 or hour >= 23)
        f['unusual_hour_score'] = int(not row['normal_active_hour_start'] <= hour <= row['normal_active_hour_end'])
        f['unusual_channel_score'] = f['preferred_channel_mismatch']
        f['unusual_location_score'] = min(10, f['distance_from_home_km']/row['normal_location_radius_km'])
        f['unusual_transaction_type_score'] = int(row['transaction_type'] != row['preferred_transaction_type'])
        f['user_behavior_score'] = min(100, 10*abs(f['amount_zscore_user']) + 10*f['unusual_location_score'] + 10*f['unusual_hour_score'])
        f.update(sequences(previous, row, f, dt))
        ah = agent[row['agent_id']]
        for suffix, duration in [('1h', 3600), ('24h', 86400)]:
            count, total, start = ah.window(t, duration) if row['agent_id'] >= 0 else (0, 0, 0)
            f[f'agent_transaction_count_{suffix}'] = count
            f[f'agent_cashout_amount_{suffix}'] = total
        f['agent_risk_score'] = min(100, f['agent_transaction_count_1h']*2 + row['blacklisted_agent']*30)
        if features is None:
            features = {key: np.empty(len(d), dtype='float32') for key in f}
        for key, value in f.items():
            features[key][i] = value
        # Mutation only after emitting all pre-event features.
        h.add(t, a, r)
        rh.add(t, a, u)
        dh.add(t, a, u)
        rel.add(t, a)
        device_users[dev].add(u)
        user_devices[u].add(dev)
        if row['agent_id'] >= 0:
            ah.add(t, a if row['transaction_type'] == 'CASH_OUT' else 0, u)
        if row['transaction_type'] in ('CASH_IN', 'REMITTANCE'):
            incoming[u].add(t, a)
        else:
            outgoing[u].add(t, a)
            if row['transaction_type'] in ('SEND_MONEY', 'BANK_TRANSFER'):
                incoming[r].add(t, a)
                graph.update(u, r)
        last[u] = {**row, **f}
    return pd.concat([d.reset_index(drop=True), pd.DataFrame(features)], axis=1)
