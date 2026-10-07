"""Measured cross-scale recommendation with explicit scientific limitations."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .config import ROOT, config, dump
from .plots import save


def compare():
    from .operating_points import operating_points
    cfg = config()
    out = ROOT/'results/final'
    out.mkdir(parents=True, exist_ok=True)
    frames, selected, meta, ablations = [], [], {}, []
    for size in cfg['sizes']:
        tag = f'{size//1000}k'
        folder = ROOT/'results'/tag
        if not (folder/'metrics.csv').exists():
            continue
        metrics = pd.read_csv(folder/'metrics.csv')
        metadata = json.loads((folder/'experiment_metadata.json').read_text())
        meta[size] = metadata
        frames.append(metrics)
        selected.append(metrics[metrics.model == metadata['selected_system']].iloc[0])
        a = pd.read_csv(folder/'ablation_study.csv')
        ablations.append(a)
    if not frames:
        raise RuntimeError('No completed experiments to compare')
    all_results = pd.concat(frames, ignore_index=True)
    best = pd.DataFrame(selected).sort_values('dataset_size')
    # Select an economical size by validation AP tolerance, never by final test AP.
    eligible = best[best.selection_pr_auc >= best.selection_pr_auc.max() - .01]
    recommended = eligible.sort_values(['dataset_size', 'training_time_seconds']).iloc[0]
    operating = operating_points()
    alternate = operating[(operating.dataset_size == recommended.dataset_size) &
                          (operating.model == recommended.model) &
                          (operating.operating_point == 'maximum_validation_f1')].iloc[0]
    all_results.to_csv(out/'all_model_results.csv', index=False)
    best.to_csv(out/'best_model_by_dataset.csv', index=False)
    best.to_csv(out/'dataset_size_comparison.csv', index=False)
    pd.concat(ablations, ignore_index=True).to_csv(out/'ablation_all_sizes.csv', index=False)
    scaling_columns = ['pr_auc', 'recall', 'precision', 'f1', 'training_time_seconds', 'p95_inference_latency_ms', 'model_file_size_mb']
    if 'stack_training_time_seconds' in all_results:
        scaling_columns.append('stack_training_time_seconds')
    for col in scaling_columns:
        for model, group in all_results.groupby('model'):
            if model in ['LogisticRegression', 'RandomForest', 'HistGradientBoosting', 'XGBoost', 'LightGBM', 'FullHybrid', 'RulesOnly']:
                plt.plot(group.dataset_size, group[col], marker='o', label=model)
        plt.xlabel('Dataset rows')
        plt.ylabel(col)
        plt.legend(fontsize=7)
        save(out, f'scale_{col}.png')
    size = int(recommended.dataset_size)
    size_tag = f'{size//1000}k'
    final_size = max(meta)
    final_tag = f'{final_size//1000}k'
    final = all_results[all_results.dataset_size == final_size].set_index('model')
    ablation = pd.read_csv(ROOT/f'results/{final_tag}/ablation_study.csv').set_index('model')
    by_type = pd.read_csv(ROOT/f'results/{final_tag}/ablation_by_fraud_type.csv')
    supervised = meta[final_size]['best_supervised']
    hybrid_delta = final.loc['FullHybrid', 'pr_auc']-final.loc[supervised, 'pr_auc']
    anomaly = ablation.loc['Hybrid_WithoutAnomaly']
    graph = ablation.loc['WithoutGraphFeatures']
    sequence = ablation.loc['WithoutSequenceFeatures']
    ablation_table = '| Variant | PR-AUC | Recall | Precision | F1 | FPR | Cost / transaction | P95 ms |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for name, row in ablation.iterrows():
        ablation_table += f'| {name} | {row.pr_auc:.4f} | {row.recall:.4f} | {row.precision:.4f} | {row.f1:.4f} | {row.false_positive_rate:.4f} | {row.cost_per_transaction:.2f} | {row.p95_inference_latency_ms:.2f} |\n'
    (out/'ablation_study.md').write_text(f'# {final_tag} ablation study\n\n'+ablation_table+'\nThresholds use validation-selected simulated minimum cost. Confidence intervals and reference models are in ablation_all_sizes.csv.\n', encoding='utf-8')
    def subtype_delta(kind, removed):
        part = by_type[by_type.fraud_type == kind].set_index('model')
        return float(part.loc[supervised, 'recall']-part.loc[removed, 'recall'])
    delta = best.iloc[-1].pr_auc-best.iloc[0].pr_auc
    recommendation = f'''# Measured synthetic benchmark recommendation

This project uses synthetic data only. Results do not represent real upay production performance.

- Recommended size: **{size_tag}**, the smallest scale within 0.01 validation PR-AUC of the best validation-selected system.
- Recommended prototype system: **{recommended.model}**. Held-out PR-AUC **{recommended.pr_auc:.4f}**, recall **{recommended.recall:.4f}**, precision **{recommended.precision:.4f}**, F1 **{recommended.f1:.4f}**.
- Synthetic binary decision threshold: **{recommended.threshold:.6f}**, selected on a disjoint validation block using assumed losses. Probability and action are separate outputs.
- The cost-optimal threshold reviews **{recommended.review_volume:.1%}** of test transactions. An alternative selected solely for validation F1 uses threshold **{alternate.threshold:.6f}**, giving test precision **{alternate.precision:.4f}**, recall **{alternate.recall:.4f}**, F1 **{alternate.f1:.4f}**, and review rate **{alternate.review_volume:.1%}**. A validation-selected 5% review-budget comparison is also saved in `operating_point_comparison.csv`; its test review rate can differ.
- Prepared-frame batch-1 P95 inference latency: **{recommended.p95_inference_latency_ms:.3f} ms**. This excludes online historical-state retrieval; production latency has not been established.
- Best supervised model by tune PR-AUC at {final_tag}: **{supervised}**.
- Full hybrid minus selected standalone supervised test PR-AUC at {final_tag}: **{hybrid_delta:+.4f}**. {'Hybrid improved this metric.' if hybrid_delta > 0 else 'Hybrid did not improve this metric.'}
- Isolation Forest incremental PR-AUC (full minus no-anomaly fusion): **{anomaly.pr_auc_delta:+.4f}**, paired test-day bootstrap 95% CI **[{anomaly.delta_ci_low:+.4f}, {anomaly.delta_ci_high:+.4f}]**. {'Positive interval supports measurable benefit on this test.' if anomaly.delta_ci_low > 0 else 'No clear positive benefit established by this interval.'}
- Graph-family retraining: selected supervised minus graph-removed PR-AUC **{graph.pr_auc_delta:+.4f}**, CI **[{graph.delta_ci_low:+.4f}, {graph.delta_ci_high:+.4f}]**; mule recall delta **{subtype_delta('MULE_ACTIVITY', 'WithoutGraphFeatures'):+.4f}**.
- Sequence-family retraining: selected supervised minus sequence-removed PR-AUC **{sequence.pr_auc_delta:+.4f}**, CI **[{sequence.delta_ci_low:+.4f}, {sequence.delta_ci_high:+.4f}]**; account-takeover recall delta **{subtype_delta('ACCOUNT_TAKEOVER', 'WithoutSequenceFeatures'):+.4f}**.
- Largest minus smallest scale test PR-AUC change for validation-selected systems: **{delta:+.4f}**. This is a descriptive population-scaling comparison, not a paired fixed-population learning curve or proof of statistical significance.
- Largest scale worthwhile under the predeclared 0.01 validation-AP trade-off rule: **{'yes' if size == final_size else 'no; a smaller scale met the rule'}**. Multiple-seed stability is not measured.

{(ROOT/f'results/{size_tag}/error_analysis.md').read_text()}

Use `{size_tag}/{recommended.model}.joblib` with its saved preprocessing, calibration and threshold for a local synthetic demo. High-risk actions route to human review. Do not use this model to make live customer decisions without real-data validation.
'''
    (out/'final_recommendation.md').write_text(recommendation, encoding='utf-8')
    sections = [
        ('Executive Summary', recommendation.split('\n', 1)[1]),
        ('Problem Definition', 'Estimate pre-transaction synthetic fraud risk, explain signals and support analyst review.'),
        ('Dataset Generation Methodology', 'Five independently generated scales with seed 42; persistent wallets and user profiles, ~80 transactions per user across 120 days. Three-event latent episodes, noisy outcomes and overlapping legitimate stress; amounts gradually inflate. No deterministic fraud labels from observed thresholds.'),
        ('Dataset Schema', 'All original mandatory columns plus historical context are retained in raw CSV and Parquet. Dataset cards and leakage inventories describe each column. Non-agent/merchant IDs use -1. Optional PageRank and some optional fine-grained telemetry are omitted; no fabricated substitutes.'),
        ('Fraud Scenarios', 'Nine fraud categories: account takeover, mule activity, velocity, subtle, device, social engineering, agent, cash-out and SIM swap. Prevalence is validated at 4–6%. Difficult scenarios intentionally overlap legitimate transactions.'),
        ('Leakage Prevention', 'Target labels/types, identifiers, rule aggregates, post-decision balances and simulator profile parameters are excluded. Prefix-invariance and target-mutation tests verify history causality. No future labels create reputation. Blacklists are fixed independently of outcomes.'),
        ('Feature Engineering', 'Rolling windows, prior means/medians/std, user-relative amounts, location distances and known-recipient history are emitted before updating state. Incoming transfers update recipient liquidity. Preprocessors are fitted only on training rows.'),
        ('Rule Engine', 'Nineteen configurable weighted rules in config.yaml. Rule-only calibrated baseline is separately evaluated. Aggregated rule score is excluded from standalone ML.'),
        ('Supervised Models', 'Logistic Regression, Random Forest, HistGradientBoosting, XGBoost and LightGBM at every scale. Imbalance uses class weights or scale_pos_weight. Seeded two-trial randomized holdout search for RF/XGB/LGBM. Secondary subtype task uses LGBM/XGB/RF with fixed CPU budgets.'),
        ('Behavioral Anomaly Detection', 'Isolation Forest fitted on legitimate training observations with at least five prior events. Cold starts explicitly abstain; global training prevalence is the fallback and fusion receives an activity flag. Anomaly raw rank is not a probability; held-out calibration maps it. Zero-observation baseline fields use simulator onboarding-profile priors rather than measured history. Replace unavailable priors with training-cohort defaults in real-data work; the fixed simulated population does not test continuous new-account arrival.'),
        ('Graph Intelligence', 'Incremental degrees, recipient concentration, shared recipient/device state, reciprocal ratios and union-find components. No all-graph PageRank recomputation. Recipient and graph information are correlated, so removal of one named family does not remove every proxy.'),
        ('Sequence Intelligence', 'Past credential/device/channel events within 24h combine with the present proposed transfer. Interpretable sequence indicators and a score replace costly recurrent networks.'),
        ('Risk Fusion', 'Logistic regression fits on the fusion validation block using raw pretrained supervised probability, anomaly rank, rules and context scores. Base calibration fitted later is deliberately not fed backward into fusion training. Fusion outputs are calibrated downstream.'),
        ('Evaluation Methodology', 'Earliest 70% train; next 15% split into five 3% blocks: tuning, fusion fitting, calibration fitting, calibration/system selection, cost threshold. Last 15% test. All test outcomes are untouched until evaluation. Returning users overlap splits by design; saved overlap and cold-start coverage quantify this.'),
        ('Results by Dataset Size', '\n'.join(f'- {int(r.dataset_size)//1000}k: {r.model}; PR-AUC {r.pr_auc:.4f}, recall {r.recall:.4f}, precision {r.precision:.4f}, F1 {r.f1:.4f}.' for r in best.itertuples())),
        ('Model Comparison', 'Full measured table: results/final/all_model_results.csv. Models and scale recommendations are selected on validation PR-AUC, not test accuracy. Failures are recorded per dataset in failures.json.'),
        ('Dataset Size Comparison', 'See dataset_size_comparison.csv and scale_*.png. Population size grows with rows, holding approximate per-user depth fixed. Single-seed, different test populations cannot establish that any scale significantly outperforms another.'),
        ('Ablation Study', 'At every scale: rules, anomaly, supervised, progressive fusion and leave-one-component-out fusion. Graph/sequence families are also removed and the selected supervised model is retrained with fixed hyperparameters. Same test rows and disjoint calibration/threshold blocks are used. Paired 95% day-cluster bootstrap intervals are conditional on fitted models; do not capture training-seed variance and are not adjusted for multiple comparisons. See ablation_all_sizes.csv and per-type recall files.\n\n'+ablation_table),
        ('Calibration', 'Identity, sigmoid and isotonic are fitted on a separate block and selected by Brier score on the select block, constrained to no more than 0.015 PR-AUC loss. Calibration curves use held-out test rows; calibration_metrics.csv records validation family selection.'),
        ('Threshold Optimization', 'Grid and validation-score quantiles select the minimum simulated cost on the threshold block. Multi-action bands minimize assumed conditional expected loss using calibrated probability; they are assumption-derived, not empirical production thresholds.'),
        ('Business Cost Analysis', 'Assumed missed-fraud loss 1000 units, false-positive investigation/friction 20 units. Binary cost and per-transaction cost are saved. Five actions use configured residual loss and legitimate costs. No real upay economics are claimed.'),
        ('Latency and Scalability', 'Single-process CPU timings include train-only preprocessing, model inference and calibration on precomputed historical frames, batches 1/10/100/1000. Repeated mean/P50/P95/P99 and throughput saved. State-service lookups and HTTP/network latency are not included. Sampled process RSS is not per-model exclusive memory. Fusion training time excludes base model training, documented in artifacts.'),
        ('Error Analysis', (ROOT/f'results/{size_tag}/error_analysis.md').read_text()+' Full counts by fraud type, transaction type, channel, amount, device, recipient, history and hour are saved.'),
        ('Adversarial Testing', 'Observed difficult fraud slices include near-threshold amounts, splitting, slow activity, known devices and normal locations. These are not an adaptive attacker simulation. A separate synthetic unseen-pattern experiment withholds SIM_SWAP_PATTERN from supervised training and every validation label fit, then reports recall; shared features do not imply zero-day generalization.'),
        ('Synthetic Segment Analysis', 'Operational segments compare recall/FPR across customer segment, account age, amount band, channel, region and rooted-device state. These analyses cannot establish real demographic fairness.'),
        ('Limitations', 'Synthetic latent scenarios, small scenario family, stationarity assumptions, one generation/training seed, correlated episodes and limited calibration support. No fixed-population sample-size learning curve; no multi-seed confidence intervals. Historical feature service and production API are not implemented. CPU budgets constrain tuning. SHAP explains uncalibrated tree margins, not causal effects or the full hybrid decision.'),
        ('Recommended Final Architecture', f'Use measured validation-selected {recommended.model} at {size_tag}, saved calibration and threshold, structured explanations and analyst review. Keep other layers only where ablation evidence supports them.'),
        ('Future Real-Data Validation Plan', 'Collect consented de-identified event histories and delayed investigation outcomes; freeze data availability times, evaluate delayed labels and user-disjoint cold starts, compare on a fixed holdout, repeat seeds, estimate actual costs, shadow deploy, monitor drift and analyst overrides before live actions.'),
        ('Conclusion', 'Artifacts reproduce a synthetic research benchmark. Measured performance and uncertainty determine the demo recommendation; no production effectiveness claim is made.'),
    ]
    report = '# MFS Guard synthetic fraud research report\n\n' + '\n\n'.join(f'## {i}. {title}\n\n{body}' for i, (title, body) in enumerate(sections, 1))
    (ROOT/'reports/final_report.md').write_text(report, encoding='utf-8')
    checks = {}
    for s in cfg['sizes']:
        tag = f'{s//1000}k'
        for path in [f'data/raw/mfs_{tag}.csv', f'data/raw/mfs_{tag}.parquet', f'results/{tag}/metrics.csv',
                     f'results/{tag}/ablation_study.csv', f'reports/dataset_cards/dataset_{tag}.md']:
            checks[path] = (ROOT/path).exists()
        checks[f'models/{tag}/trained_artifacts'] = len(list((ROOT/'models'/tag).glob('*.joblib'))) >= 8
    for path in ['results/final/all_model_results.csv', 'results/final/dataset_size_comparison.csv',
                 'results/final/final_recommendation.md', 'reports/final_report.md']:
        checks[path] = (ROOT/path).exists()
    dump(out/'mandatory_output_check.json', checks)
    dump(out/'recommendation.json', recommended.to_dict())
    print(f'MFS GUARD ML EXPERIMENTS: {len(frames)}/5 SCALES COMPLETE\nRecommended: {size_tag} {recommended.model}\nTest PR-AUC: {recommended.pr_auc:.4f}\nReport: {ROOT / "reports/final_report.md"}', flush=True)
    return checks
