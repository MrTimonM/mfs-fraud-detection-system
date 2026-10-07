"""Validation-selected alternatives expose the recall/review-cost trade-off."""
import numpy as np
import pandas as pd
from .config import ROOT, config
from .evaluate import metrics


def operating_points():
    cfg = config()
    records = []
    for size in cfg['sizes']:
        folder = ROOT/f'results/{size//1000}k'
        if not (folder/'experiment_metadata.json').exists():
            continue
        curves = pd.read_csv(folder/'threshold_metrics.csv')
        predictions = pd.read_parquet(folder/'test_predictions.parquet')
        local = []
        for model, curve in curves.groupby('model'):
            test = predictions[predictions.model_name == model]
            y, p = test.fraud_label.to_numpy(), test.predicted_probability.to_numpy()
            options = {
                'minimum_simulated_cost': curve.sort_values(['expected_business_cost', 'false_positive_rate']).iloc[0],
                'maximum_validation_f1': curve.sort_values(['f1', 'precision'], ascending=False).iloc[0],
                'validation_review_budget_5pct': curve[curve.review_volume <= .05].sort_values(['recall', 'precision'], ascending=False).iloc[0],
            }
            for name, selected in options.items():
                local.append({'dataset_size': size, 'model': model, 'operating_point': name,
                    'validation_review_volume': selected.review_volume, 'validation_f1': selected.f1,
                    **metrics(y, p, float(selected.threshold), cfg),
                    'selection_scope': 'threshold validation block only; test review rate may shift'})
        pd.DataFrame(local).to_csv(folder/'operating_point_comparison.csv', index=False)
        records.extend(local)
    table = pd.DataFrame(records)
    table.to_csv(ROOT/'results/final/operating_point_comparison.csv', index=False)
    return table
