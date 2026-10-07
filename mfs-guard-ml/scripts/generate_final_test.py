"""STEP 1: one untouched common benchmark generated with a never-used seed.

Reuses the training generator, validation and leakage audit unchanged. The
generator is called with save=False so no training artifact is overwritten.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import pyarrow.parquet as pq
from src.config import ROOT, config, dump
from src.data_generator import generate
from src.data_validation import validate
from src.fraud_scenarios import FRAUD_TYPES
from src.leakage_audit import audit, feature_columns

FINAL_TEST_SEED = 2026
OUT = ROOT/'data/final_test'


def used_seeds():
    seeds = {json.loads(p.read_text())['seed'] for p in (ROOT/'data/metadata').glob('generation_*.json')}
    return seeds | {42, 123}  # 42: default/tests, 123: reproducibility test


def pct(series):
    return {str(k): {'count': int(v), 'percent': round(100*v/series.sum(), 3)} for k, v in series.items()}


def card(d, info):
    lines = ['# MFS Guard common final test set (100k)', '',
        'Synthetic data only. Results measured on it do not represent real upay production performance.', '',
        f'Generated with the unchanged training generator (`src/data_generator.py`), seed **{info["seed"]}** '
        f'(never used for training, validation, tuning or tests; previously used seeds: {info["previously_used_seeds"]}). '
        'Histories are a self-contained 120-day chronological replay; every feature is computed from strictly earlier '
        'events of this set only. No training/validation row or transaction ID is reused.', '',
        '## Summary', '', '| Item | Value |', '|---|---:|']
    for key in ['rows', 'users', 'receivers', 'devices', 'agents', 'merchants', 'columns', 'model_feature_count',
                'fraud_count', 'fraud_prevalence_percent', 'cold_start_percent', 'new_device_percent',
                'new_recipient_percent', 'night_transaction_percent', 'high_velocity_percent']:
        lines.append(f'| {key} | {info[key]} |')
    lines += ['', f'Time range: {info["time_range"][0]} to {info["time_range"][1]}', '',
        'Definitions: cold start = fewer than five prior customer events (`behavioral_model_active == 0`, the anomaly '
        'layer abstains); new device = device not previously seen for that customer; new recipient = first transfer '
        'to that receiver; high velocity = three or more prior customer events in the previous five minutes.', '']
    for title, key in [('Fraud-type distribution', 'fraud_types'), ('Channel distribution', 'channels'),
                       ('Transaction-type distribution', 'transaction_types')]:
        lines += [f'## {title}', '', '| Value | Count | % |', '|---|---:|---:|']
        lines += [f'| {k} | {v["count"]} | {v["percent"]} |' for k, v in info[key].items()]
        lines.append('')
    lines += ['## Fraud recall-relevant slices (fraud rows)', '', '| Slice | Fraud rows |', '|---|---:|']
    lines += [f'| {k} | {v} |' for k, v in info['fraud_slices'].items()]
    lines += ['', '## Missing values', '',
        f'Columns with any missing value: {info["missing"]["columns_with_missing"] or "none"}. '
        f'Total missing cells: {info["missing"]["total_missing_cells"]}. Absent agent/merchant identifiers are encoded as -1 by design.', '',
        '## Validation', '',
        f'- Schema, accounting, chronology and sampled history recomputation: **{info["validation"]["status"]}** '
        f'({info["validation"]["sampled_history_checks"]} sampled history checks)',
        f'- Leakage audit: **{info["leakage"]["status"]}**; maximum absolute feature/target correlation '
        f'{info["leakage"]["max_absolute_target_correlation"]:.4f}',
        f'- Transaction-ID overlap with all training datasets: **{info["id_overlap_with_training"]}**',
        f'- All supported fraud categories present: **{info["all_fraud_types_present"]}**', '',
        '## Limitations', '',
        'The population is new (different seed) but drawn from the same simulator as training, so this measures '
        'generalisation to fresh synthetic customers and episodes, not to real behaviour or new fraud mechanics. Unlike the '
        'per-scale test splits (last 15% of a 120-day replay with warm history), this set includes the cold-start warm-up '
        'period, so absolute metrics are not directly comparable with earlier per-scale results.', '']
    return '\n'.join(lines)


def main(seed):
    seeds = used_seeds()
    if seed in seeds:
        raise SystemExit(f'Seed {seed} was already used: {sorted(seeds)}')
    OUT.mkdir(parents=True, exist_ok=True)
    rows = 100000
    d = generate(rows, seed, save=False)
    validation = validate(d, rows)
    leakage = audit(d)
    # Critical checks beyond the generator's own validation.
    training_ids = set()
    for path in sorted((ROOT/'data/raw').glob('mfs_*k.parquet')):
        training_ids.update(pq.read_table(path, columns=['transaction_id']).column(0).to_pylist())
    overlap = int(d.transaction_id.isin(training_ids).sum())
    missing_types = sorted(set(FRAUD_TYPES) - set(d.fraud_type))
    critical = {'id_overlap': overlap == 0, 'all_fraud_types': not missing_types,
                'prevalence_4_6pct': .04 <= d.fraud_label.mean() <= .06,
                'validation': validation['status'] == 'PASSED', 'leakage': leakage['status'] == 'PASSED'}
    fraud = d[d.fraud_label == 1]
    info = {'seed': seed, 'previously_used_seeds': sorted(seeds), 'rows': len(d),
        'users': int(d.user_id.nunique()), 'receivers': int(d.receiver_id.nunique()),
        'devices': int(d.device_id.nunique()), 'agents': int(d.loc[d.agent_id >= 0, 'agent_id'].nunique()),
        'merchants': int(d.loc[d.merchant_id >= 0, 'merchant_id'].nunique()), 'columns': len(d.columns),
        'model_feature_count': len(feature_columns(d)), 'fraud_count': int(d.fraud_label.sum()),
        'fraud_prevalence_percent': round(100*d.fraud_label.mean(), 3),
        'cold_start_percent': round(100*(d.behavioral_model_active == 0).mean(), 3),
        'new_device_percent': round(100*d.device_is_new.mean(), 3),
        'new_recipient_percent': round(100*d.first_time_recipient.mean(), 3),
        'night_transaction_percent': round(100*d.is_night_transaction.mean(), 3),
        'high_velocity_percent': round(100*(d.tx_count_5m >= 3).mean(), 3),
        'time_range': [str(d.timestamp.min()), str(d.timestamp.max())],
        'fraud_types': pct(d.fraud_type.value_counts()), 'channels': pct(d.channel.value_counts()),
        'transaction_types': pct(d.transaction_type.value_counts()),
        'fraud_slices': {'cold_start': int((fraud.behavioral_model_active == 0).sum()),
            'new_device': int(fraud.device_is_new.sum()), 'known_device': int((fraud.device_is_new == 0).sum()),
            'first_time_recipient': int(fraud.first_time_recipient.sum()),
            'known_recipient': int((fraud.first_time_recipient == 0).sum()),
            'high_velocity': int((fraud.tx_count_5m >= 3).sum()),
            'sim_changed_recently': int(fraud.sim_changed_recently.sum())},
        'missing': {'columns_with_missing': {c: int(v) for c, v in d.isna().sum().items() if v},
                    'total_missing_cells': int(d.isna().sum().sum())},
        'validation': {k: validation[k] for k in ('status', 'sampled_history_checks', 'duplicates')},
        'leakage': {k: leakage[k] for k in ('status', 'max_absolute_target_correlation')},
        'id_overlap_with_training': overlap, 'training_ids_checked': len(training_ids),
        'all_fraud_types_present': not missing_types, 'critical_checks': critical,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'generator': 'src/data_generator.generate (unchanged)', 'config': config()}
    dump(OUT/'final_test_metadata.json', info)
    (ROOT/'reports').mkdir(exist_ok=True)
    (ROOT/'reports/final_test_dataset_card.md').write_text(card(d, info), encoding='utf-8')
    if not all(critical.values()):
        raise SystemExit(f'CRITICAL VALIDATION FAILED: {critical}. Dataset not saved.')
    d.to_parquet(OUT/'mfs_final_test_100k.parquet', index=False)
    d.to_csv(OUT/'mfs_final_test_100k.csv', index=False)
    print(f'Final test set: {len(d):,} rows, prevalence {d.fraud_label.mean():.3%}, '
          f'{info["cold_start_percent"]}% cold start; all critical checks passed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=FINAL_TEST_SEED)
    main(parser.parse_args().seed)
