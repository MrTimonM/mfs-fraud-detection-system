"""Headless, reproducible experiment figures."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve


def save(out, name):
    plt.tight_layout()
    plt.savefig(out/name, dpi=125, bbox_inches='tight')
    plt.close('all')


def plots(d, y, scores, metrics, out, thresholds):
    for column, name in [('fraud_label', 'class_distribution'), ('fraud_type', 'fraud_type_distribution')]:
        d[column].value_counts().plot.bar(figsize=(9, 4))
        save(out, name+'.png')
    for label, group in d.groupby('fraud_label'):
        plt.hist(np.log10(np.maximum(group.amount, .01)), bins=50, alpha=.5, density=True, label=f'label {label}')
    plt.xlabel('log10 amount (BDT)')
    plt.legend()
    save(out, 'amount_distribution.png')
    for kind in ('roc', 'pr'):
        plt.figure(figsize=(9, 6))
        for name, p in scores.items():
            if name.startswith('Without') or name.startswith('Hybrid_'):
                continue
            if kind == 'roc':
                x, yy, _ = roc_curve(y, p)
            else:
                yy, x, _ = precision_recall_curve(y, p)
            plt.plot(x, yy, label=name)
        plt.xlabel('False positive rate' if kind == 'roc' else 'Recall')
        plt.ylabel('True positive rate' if kind == 'roc' else 'Precision')
        plt.legend(fontsize=7)
        save(out, 'roc_comparison.png' if kind == 'roc' else 'pr_curve_comparison.png')
    for name, p in scores.items():
        matrix = confusion_matrix(y, p >= thresholds[name], labels=[0, 1])
        ConfusionMatrixDisplay(matrix).plot()
        plt.title(name)
        save(out, f'confusion_matrix_{name}.png')
    plt.figure(figsize=(8, 6))
    for name in list(scores)[:7] + (['FullHybrid'] if 'FullHybrid' in scores else []):
        true, pred = calibration_curve(y, scores[name], n_bins=10, strategy='quantile')
        plt.plot(pred, true, marker='.', label=name)
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.legend(fontsize=7)
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Observed fraud fraction')
    save(out, 'calibration_curve.png')
    for col, name in [('training_time_seconds', 'training_time_comparison.png'),
                      ('p95_inference_latency_ms', 'inference_latency_comparison.png')]:
        metrics.set_index('model')[col].plot.bar(figsize=(10, 5))
        plt.ylabel(col)
        save(out, name)
