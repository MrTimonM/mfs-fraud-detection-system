"""SHAP attributions computed from the trained tree, with local signed evidence."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .config import dump


def explain(model, x, names, out, transaction_ids, local_x=None, local_ids=None):
    import shap
    values = np.asarray(shap.TreeExplainer(model).shap_values(x[:256]))
    if values.ndim == 3:
        values = values[:, :, 1]
    importance = pd.DataFrame({'feature': names, 'mean_absolute_shap': np.abs(values).mean(axis=0)})
    importance.sort_values('mean_absolute_shap', ascending=False).to_csv(out/'shap_feature_importance.csv', index=False)
    shap.summary_plot(values, x[:256], feature_names=list(names), show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(out/'shap_summary.png', dpi=130, bbox_inches='tight')
    plt.close('all')
    if local_x is not None:
        values = np.asarray(shap.TreeExplainer(model).shap_values(local_x))
        if values.ndim == 3:
            values = values[:, :, 1]
        x, transaction_ids = local_x, local_ids
    locals_ = []
    for i in range(min(12, len(values))):
        order = np.argsort(values[i])
        locals_.append({'transaction_id': transaction_ids[i], 'explanation_scale': 'uncalibrated tree raw margin',
            'top_positive_risk_factors': [{'feature': names[j], 'value': float(x[i, j]), 'shap': float(values[i, j])} for j in order[-5:][::-1]],
            'top_negative_risk_factors': [{'feature': names[j], 'value': float(x[i, j]), 'shap': float(values[i, j])} for j in order[:5]]})
    dump(out/'local_explanations.json', locals_)
    return importance
