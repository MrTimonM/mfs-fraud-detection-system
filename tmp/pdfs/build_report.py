from pathlib import Path
import json, math, hashlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT
from xml.sax.saxutils import escape

ROOT=Path(__file__).resolve().parents[2]
ML=ROOT/'mfs-guard-ml'
TMP=ROOT/'tmp/pdfs'
OUT=ROOT/'output/pdf'
OUT.mkdir(parents=True,exist_ok=True)
R=ML/'results/final_common_test'
df=pd.read_csv(R/'model_comparison.csv')
chosen=df[df.model=='LightGBM_200k'].iloc[0]
ci=pd.read_csv(R/'confidence_intervals.csv')
types=pd.read_csv(R/'fraud_type_performance.csv').query("model == 'LightGBM_200k'")
cost=pd.read_csv(R/'business_cost_comparison.csv').query("model == 'LightGBM_200k'")
meta=json.loads((ML/'models/200k/LightGBM_metadata.json').read_text())
features=json.loads((ML/'backend/models/final/feature_list.json').read_text())
NAVY='#142D42'; TEAL='#007E87'; ORANGE='#D37335'; GREY='#607486'; PALE='#EAF2F4'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':NAVY,'text.color':NAVY,'xtick.color':GREY,'ytick.color':NAVY,'axes.edgecolor':'#CCD7DF','figure.facecolor':'white','savefig.facecolor':'white'})
def save(fig,name):
    fig.savefig(TMP/(name+'.png'),dpi=190,bbox_inches='tight',pad_inches=.18)
    plt.close(fig)
def short(s):
    return s.replace('Hybrid_WithoutSequence','Hybrid - sequence').replace('Hybrid_AddAnomaly','Rules + ML + anomaly').replace('Hybrid_RulesSupervised','Rules + ML').replace('FullHybrid','Full hybrid').replace('RulesOnly','Rules only').replace('IsolationForest','Isolation Forest').replace('_',' / ')
def bars(data,col,name,xlabel):
    d=data.sort_values(col)
    fig,ax=plt.subplots(figsize=(9,4.5))
    b=ax.barh([short(x) for x in d.model],d[col],color=[TEAL if x=='LightGBM_200k' else '#A8BBC7' for x in d.model],height=.65)
    ax.bar_label(b,fmt='%.4f' if col=='pr_auc' else '%.2f',padding=5,fontsize=9)
    ax.set_xlim(0,d[col].max()*1.20); ax.set_xlabel(xlabel); ax.grid(axis='x',alpha=.15); ax.set_axisbelow(True)
    fig.tight_layout(); save(fig,name)
bars(df,'pr_auc','comparison','Average precision (reported as PR-AUC); higher is better')
bars(df,'p95_latency_ms','latency','Prepared-frame batch-1 P95 latency (ms); lower is better')
fig,ax=plt.subplots(figsize=(9,3.8))
d=types.sort_values('count'); b=ax.barh(d.fraud_type.str.replace('_',' ').str.title(),d['count'],color=TEAL)
ax.bar_label(b,padding=4);ax.set_xlim(0,1100);ax.set_xlabel('Fraud transactions in the common 100k test');save(fig,'population')
fig,ax=plt.subplots(figsize=(9,3.9))
d=types.sort_values('recall_at_recommended_threshold'); y=np.arange(len(d))
ax.barh(y-.17,d.recall_at_recommended_threshold,height=.32,label='Selected threshold (4.10% flagged)',color=TEAL)
ax.barh(y+.17,d.recall_at_cost_optimal_threshold,height=.32,label='Cost reference (33.18% flagged)',color='#B9CDD7')
ax.set_yticks(y,d.fraud_type.str.replace('_',' ').str.title());ax.xaxis.set_major_formatter(PercentFormatter(1));ax.set_xlim(0,1);ax.legend(loc='lower right',fontsize=8);ax.set_xlabel('Recall within fraud category');save(fig,'types')
sw=pd.read_csv(R/'threshold_comparison.csv').query("model == 'LightGBM_200k'").sort_values('threshold')
fig,ax=plt.subplots(figsize=(9,3.8))
for k,c,l in [('precision',TEAL,'Precision'),('recall',ORANGE,'Recall'),('review_rate',GREY,'Flagged share')]:ax.plot(sw.threshold,sw[k],label=l,color=c,lw=2.3)
ax.axvline(chosen.recommended_threshold,color=NAVY,ls='--',label='Selected: 0.20517');ax.set_xlim(.05,.60);ax.set_ylim(0,.7);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_xlabel('Probability threshold');ax.legend(fontsize=9);save(fig,'threshold')
fig,axes=plt.subplots(1,2,figsize=(9,3.6),gridspec_kw={'width_ratios':[1,1.15]})
cm=np.array([[92191,2586],[3705,1518]])
axes[0].imshow(np.log10(cm+1),cmap='Blues')
for i in range(2):
    for j in range(2):axes[0].text(j,i,f'{cm[i,j]:,}',ha='center',va='center',fontsize=17,color='white' if cm[i,j]>10000 else NAVY)
axes[0].set_xticks([0,1],['Not flagged','Flagged']);axes[0].set_yticks([0,1],['Legitimate','Fraud']);axes[0].set_title('100,000 common-test transactions')
axes[1].axis('off');axes[1].text(.05,.88,'At threshold 0.20517',fontsize=16,weight='bold')
axes[1].text(.05,.71,'1,518 fraud transactions caught\n3,705 fraud transactions missed\n2,586 legitimate transactions flagged',fontsize=12,linespacing=1.8,va='top')
axes[1].text(.05,.13,'36.99% precision   |   29.06% recall\n4.10% flagged   |   2.73% false-positive rate',fontsize=11,linespacing=1.7);save(fig,'confusion')
fig,ax=plt.subplots(figsize=(9,3.8))
sd=pd.read_csv(ML/'results/final/dataset_size_comparison.csv')
ax.plot(sd.dataset_size/1000,sd.pr_auc,'o-',lw=2,color=TEAL)
for x,y in zip(sd.dataset_size/1000,sd.pr_auc):ax.annotate(f'{y:.4f}',(x,y),xytext=(0,9),textcoords='offset points',ha='center')
ax.set_ylim(.19,.265);ax.set_xticks([100,200,300,400,500]);ax.set_xlabel('Generated dataset size (thousands)');ax.set_ylabel('Own temporal-test average precision');ax.grid(alpha=.15);save(fig,'scales')
sh=pd.read_csv(ML/'results/200k/shap_feature_importance.csv').head(10).iloc[::-1]
fig,ax=plt.subplots(figsize=(9,3.9));ax.barh(sh.feature.str.replace('_',' '),sh.mean_absolute_shap,color=TEAL);ax.set_xlabel('Mean absolute SHAP attribution (uncalibrated model output)');save(fig,'shap')
# Architecture illustration: explicit exported path and parallel policy signals.
from matplotlib.patches import FancyBboxPatch
fig,ax=plt.subplots(figsize=(10,4.5));ax.set_xlim(0,10);ax.set_ylim(0,4.5);ax.axis('off')
def box(x,y,w,h,title,detail,fc=PALE):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08,rounding_size=0.12',facecolor=fc,edgecolor='#B6C9D4'))
    ax.text(x+w/2,y+h*.65,title,ha='center',va='center',fontsize=11,weight='bold');ax.text(x+w/2,y+h*.29,detail,ha='center',va='center',fontsize=8.8)
def arrow(x,y,xx,yy):ax.annotate('',(xx,yy),(x,y),arrowprops={'arrowstyle':'->','color':GREY,'lw':1.7})
box(.1,2.5,2,.95,'Proposed transfer','Amount / channel / device')
box(2.65,2.5,2.15,.95,'Historical features','130 input columns')
box(5.35,2.5,1.9,.95,'LightGBM','140 boosted trees')
box(7.8,2.5,2,.95,'Sigmoid calibration','Fraud probability')
for a,b in [(2.15,2.55),(4.85,5.25),(7.3,7.7)]:arrow(a,2.98,b,2.98)
box(2.65,.5,2.15,1,'19 weighted rules','Separate evidence')
box(5.35,.5,1.9,1,'Isolation Forest','Supporting anomaly rank')
box(7.8,.5,2,1,'Decision + evidence','Monitor / step-up / hold')
arrow(3.7,2.4,3.7,1.6);arrow(4.6,2.4,6.1,1.6);arrow(7.35,1,7.7,1);arrow(8.8,2.4,8.8,1.6)
ax.plot([3.7,3.7,8.8],[.4,.15,.15],color=GREY,lw=1.5);arrow(8.8,.15,8.8,.4)
ax.text(.15,.65,'SERVING DESIGN\nML service required\nRules fallback if offline',fontsize=10,color=ORANGE,linespacing=1.5)
save(fig,'architecture')

for name,file in [('Body','arial.ttf'),('Bold','arialbd.ttf'),('Italic','ariali.ttf')]:pdfmetrics.registerFont(TTFont(name,str(Path('C:/Windows/Fonts')/file)))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='Bold',italic='Italic',boldItalic='Bold')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='Title2',fontName='Bold',fontSize=28,leading=32,textColor=colors.HexColor(NAVY),spaceAfter=16))
styles.add(ParagraphStyle(name='Deck',fontName='Body',fontSize=12,leading=17,textColor=colors.HexColor(GREY),spaceAfter=13))
styles.add(ParagraphStyle(name='Text2',fontName='Body',fontSize=10,leading=14.2,textColor=colors.HexColor(NAVY),spaceAfter=9))
styles.add(ParagraphStyle(name='Sub2',fontName='Bold',fontSize=13,leading=17,textColor=colors.HexColor(TEAL),spaceBefore=9,spaceAfter=7))
styles.add(ParagraphStyle(name='Small2',fontName='Body',fontSize=8,leading=10.5,textColor=colors.HexColor(GREY),spaceAfter=6))
styles.add(ParagraphStyle(name='Cell2',fontName='Body',fontSize=8,leading=10.5,textColor=colors.HexColor(NAVY)))
styles.add(ParagraphStyle(name='Feature',fontName='Body',fontSize=8,leading=11,textColor=colors.HexColor(NAVY)))
story=[]
def p(t,style='Text2'):return Paragraph(t,styles[style])
def add(t,style='Text2'):story.append(p(t,style))
def title(k,t,deck):
    if story:story.append(PageBreak())
    add(k.upper(),'Small2');add(t,'Title2');add(deck,'Deck')
def sub(t):add(t,'Sub2')
def source(t):story.append(Spacer(1,8));add('Evidence: '+t,'Small2')
def img(name,w=500,h=None):
    from PIL import Image as PI
    path=TMP/(name+'.png');iw,ih=PI.open(path).size
    story.append(Image(str(path),width=w,height=h or w*ih/iw));story.append(Spacer(1,8))
def table(headers,rows,widths=None,size=8):
    if widths and sum(widths)>500: widths=[w*500/sum(widths) for w in widths]
    vals=[[p('<b>'+escape(str(c))+'</b>','Cell2') for c in headers]]+[[p(escape(str(c)),'Cell2') for c in row] for row in rows]
    t=Table(vals,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#DBE9EE')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F3F7F9')]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#B8CDD6'))]))
    story.append(t);story.append(Spacer(1,8))
def pct(x):return f'{x*100:.2f}%'

title('MFS GUARD / JUDGES TECHNICAL BRIEF','Fraud detection.\nEvidence before complexity.','Model findings, comparisons and technical implementation | 07 October 2026')
add('Selected exported model: <b>LightGBM_200k-v1</b>','Sub2')
add('A compact, calibrated fraud-ranking model supported by interpretable rules and behavioral signals. The latest saved selection uses LightGBM from the 200k experiment, with sigmoid calibration and 130 input features.')
table(['0.2417 PR-AUC','10.90 ms P95','4.10% flagged'],[['Common synthetic test','Prepared-frame, batch 1','At selected threshold']], [167,167,166])
img('architecture')
add('<b>Scope:</b> 100,000 fresh synthetic test transactions; 5,223 fraud cases. These are simulator findings, not real customer or upay production results. ML serving is configurable; when unavailable, the app falls back to rules. Benchmark results measure the saved model, not the complete app policy.','Small2')
source('backend/models/final/model_metadata.json; results/final_common_test/model_comparison.csv; src/lib/engine.ts (repository root).')

title('01 / Decision','Which model do we use?','Use the latest exported LightGBM model for the ML prototype; keep the evidence and operational policy separate.')
table(['Decision factor','LightGBM 200k','Hybrid without sequence 500k'],[
['Common-test PR-AUC','0.24168','0.24153'],['95% paired AP difference','Reference','-0.00624 to +0.00510 vs LightGBM'],['Batch-1 P95','10.90 ms','21.45 ms'],['Saved experiment bundle','0.113 MiB','2.037 MiB'],['At selected thresholds: F1','0.3255','0.3307'],['At selected thresholds: recall','29.06%','30.17%']], [190,155,155])
sub('Why this selection is defensible')
add('LightGBM achieves almost the same ranking quality as the strongest hybrid with about half its measured P95 inference time and an approximately 18 times smaller saved bundle. The near-zero AP difference does <b>not</b> establish statistically superior detection. The hybrid has slightly higher F1 and recall at its own validation-selected threshold.')
sub('Resolve the older recommendation')
add('The earlier per-scale report recommended 100k Hybrid_WithoutGraph using a validation-AP tolerance rule. The later common-test selection file and exported metadata name LightGBM_200k. This report uses the later exported artifact as the answer to “which model we use,” while retaining earlier findings as separate experiments.')
sub('Selection discipline')
add('Final thresholds were selected on each training run\'s validation block. The repository does not establish that the final model choice was frozen before viewing common-test results. Treat the common set as comparative evidence; reserve another untouched holdout for confirmation after freezing the selection.')
source('final_common_test/final_selection.json, confidence_intervals.csv, model_comparison.csv; reports/final_report.md. All paths in this report are relative to mfs-guard-ml unless stated otherwise.')

title('02 / Test population','What did we evaluate?','Fresh simulated customers and episodes, drawn from the same generator as training.')
table(['Population','Common final test'],[['Transactions / fraud','100,000 / 5,223 (5.223%)'],['Users / devices / agents','1,250 / 1,666 / 83'],['Replay / seed','120-day simulation / seed 2026'],['Cold starts / new devices','6.25% / 8.775% of transactions'],['Schema / inputs','154 columns / 130 model input columns'],['Transaction-ID overlap','0 with training datasets']], [230,270])
img('population',480)
add('The generator includes noisy outcomes, persistent wallets, multi-event fraud episodes, legitimate stress behavior and overlapping fraud signals. APP, AGENT, USSD, WEB and API channels are represented. All nine fraud categories occur in the common test.','Small2')
source('reports/final_test_dataset_card.md; src/data_generator.py; results/final_common_test/fraud_type_performance.csv.')

title('03 / Experimental design','Chronological, separated evaluation','Five generated scales: 100k, 200k, 300k, 400k and 500k. Training seed 42; fresh common-test seed 2026.')
table(['200k run block','Share','Rows','Purpose'],[['Train','70%','140,000','Fit preprocessing and base estimators'],['Tune','3%','6,000','Choose hyperparameters by AP'],['Fusion','3%','6,000','Fit logistic stacking model'],['Calibrate','3%','6,000','Fit calibration candidates'],['Select','3%','6,000','Choose calibration / system'],['Threshold','3%','6,000','Choose action operating points'],['Own temporal test','15%','30,000','Evaluate returning customers']], [110,45,65,280])
sub('Leakage controls')
add('Features are emitted from prior history before state updates. Targets, fraud types, identifiers, post-decision balances, rule aggregates and simulator profile parameters are excluded from standalone supervised inputs. Imputation, scaling and categorical encoding are fitted on training rows only. Prefix-invariance and target-mutation tests are used to check causal history construction.')
sub('Two different evaluation questions')
add('The per-scale 15% tests evaluate later transactions from returning customers (100% of 200k test users were seen in training). The common test replays fresh synthetic histories and includes the cold-start warm-up. Their absolute metrics should not be mixed or presented as one learning curve.')
sub('What is verified in saved evidence')
add('The common-test dataset card records schema/accounting/chronology checks, 96 sampled history checks, a passed leakage audit, no missing cells and no transaction-ID overlap. These are saved findings; this report does not claim a new training or test-suite run.')
source('results/200k/split_boundaries.json; src/preprocessing.py; reports/final_test_dataset_card.md; reports/final_report.md.')

title('04 / Common-test comparison','Ranking quality across finalists','Every bar uses the same 100,000 synthetic transactions. Higher average precision is better.')
img('comparison',500)
add('The experiment labels average precision as “PR-AUC.” The prevalence reference is <b>0.05223</b>; LightGBM\'s AP is about <b>4.63 times</b> that value. This is a ranking-quality comparison, not a probability of catching all fraud.')
add('LightGBM 200k (0.24168), Hybrid without sequence 500k (0.24153), Hybrid + anomaly 500k (0.24131) and Rules + ML 500k (0.24126) form a tight cluster. More layers do not automatically yield better ranking. Rules alone and Isolation Forest alone are substantially weaker here.')
table(['Selected model metric','Estimate','95% day-bootstrap interval'],[[x.replace('_',' ').upper(),f'{chosen[x]:.4f}',f'{ci[(ci.model=="LightGBM_200k") & (ci.metric==x)].iloc[0].ci_low:.4f} to {ci[(ci.model=="LightGBM_200k") & (ci.metric==x)].iloc[0].ci_high:.4f}'] for x in ['pr_auc','precision','recall','f1']], [170,100,230])
source('final_common_test/model_comparison.csv; confidence_intervals.csv. 1,000 UTC-day bootstrap resamples, conditional on fitted models; no training-seed uncertainty.')

title('05 / Operating-point comparison','The complete model scorecard','Precision, recall, F1 and flagged share use each model\'s own maximum-validation-F1 threshold.')
table(['Model / training scale','AP','ROC AUC','Precision','Recall','F1','Flagged'],[[short(r.model),f'{r.pr_auc:.4f}',f'{r.roc_auc:.4f}',pct(r.precision),pct(r.recall),f'{r.f1:.4f}',pct(r.review_volume)] for r in df.itertuples()],[155,48,57,62,59,50,69])
sub('How to read the metrics')
add('<b>Precision</b> = caught fraud / all flagged transactions. <b>Recall</b> = caught fraud / all actual fraud. <b>F1</b> balances the two. <b>False-positive rate</b> = flagged legitimate / all legitimate. <b>AP</b> summarizes precision over recall as the threshold varies; <b>ROC AUC</b> measures positive-versus-negative ranking.')
add('The selected model has MCC <b>0.2954</b>, balanced accuracy <b>0.6317</b>, false-positive rate <b>2.7285%</b>, false-negative rate <b>70.9362%</b> and Brier score <b>0.04399</b>. Accuracy alone is misleading: predicting every transaction as legitimate already yields 94.777% accuracy on this population.')
sub('Fair budget comparison')
table(['LightGBM ranked review budget','Recall','Precision'],[['Top 1%',pct(chosen.recall_at_1pct_review_budget),pct(chosen.precision_at_1pct_review_budget)],['Top 2%',pct(chosen.recall_at_2pct_review_budget),pct(chosen.precision_at_2pct_review_budget)],['Top 5%',pct(chosen.recall_at_5pct_review_budget),pct(chosen.precision_at_5pct_review_budget)]],[260,120,120])
add('Top-budget results rank the test scores and use fractional random tie-breaking at the boundary. They describe queue capacity; they are not newly tuned deployable probability thresholds.','Small2')
source('final_common_test/model_comparison.csv; src/common_test.py (budget_recall).')

title('06 / Errors at the selected threshold','What the chosen model misses','A useful judge-facing claim must include both detections and misses.')
img('confusion',500)
img('types',480)
add('<b>Main weakness:</b> subtle fraud recall is 0.99%, velocity fraud 0.98% and agent fraud 1.62% at the selected threshold. Account takeover is stronger at 65.94%. Subtle fraud contributes 897 missed cases. The lower cost-reference threshold improves recall while flagging one-third of all transactions.','Small2')
source('final_common_test/model_comparison.csv; fraud_type_performance.csv. Matrix values are counts; cell shading uses a log scale for readability.')

title('07 / Thresholds and actions','Probability is not the action','Calibrated probability, weighted rules and operational capacity are separate decisions.')
img('threshold',490)
table(['Exported ML band','Threshold','Intended tier'],[['Below monitor','p < 0.082229','Approve'],['Monitor','0.082229 <= p < 0.205167','Approve and monitor'],['Step-up','0.205167 <= p < 0.445','Additional authentication'],['Hold','p >= 0.445','Temporary hold / review']],[130,180,190])
add('The step-up threshold <b>0.20516681</b> maximizes F1 on the original validation threshold block. Monitor and hold are chosen from validation budgets of 15% and 1%. These budgets are not guarantees on future data. On the common test, 7,506 are monitor-only, 3,337 step-up and 767 hold.')
add('The exported policy also records rule-score thresholds 20 / 40 / 70 and a reject condition requiring both ML at least hold and rule score at least 70. The benchmarked tiered policy measures monitor / step-up / hold only; it does not measure that combined reject rule. Sweeps above are descriptive and must not be used to retune on this test.','Small2')
source('backend/models/final/thresholds.json; final_common_test/threshold_comparison.csv, business_cost_comparison.csv; src/common_test.py.')

title('08 / Cost and workload','Higher recall has a workload price','All costs are assumed simulation units. They are not measured savings or real provider economics.')
table(['Policy on common test','Flagged / interrupted','Recall','Estimated cost'],[['Selected validation tiers','4,104 (4.10%)','29.06% at step-up','3,834,605'],['Lower binary cost reference','33,178 (33.18%)','70.69%','2,667,786'],['Approve everything (derived)','0','0%','5,223,000']],[190,120,90,100])
sub('Assumptions behind the figures')
add('Each fraud case carries 1,000 loss units before intervention. Assumed residual fraud loss: approve 100%, monitor 80%, step-up 25%, hold 5%. Step-up costs 2 units per affected transaction; hold costs 8. Interrupted legitimate users incur 5 friction units; held legitimate users also incur 15 investigation units.')
add('The selected tiered policy\'s 3,834,605 total consists of 3,802,700 residual fraud-loss units, 19,095 false-positive costs, 6,674 verification costs and 6,136 hold costs. The code does not add a separate monitor operating fee in this common-test calculation.')
sub('Avoid confusing two cost functions')
add('The lower threshold 0.03734065 was selected using the training-run binary objective <b>1,000 x false negatives + 20 x false positives</b>. The common-test policy evaluation then applies the action-specific residual losses above. “Cost-optimal” is a reference to the original validation objective, not proof that it is globally optimal for the tiered policy.')
sub('A defensible operating choice')
add('The selected threshold controls interruption volume but misses most fraud. A real deployment would need actual fraud loss, verification efficacy, analyst capacity and customer friction estimates before choosing a policy. The lower threshold\'s 11.13% precision illustrates the cost of increased recall.')
source('config.yaml; src/common_test.py (business_cost); final_common_test/business_cost_comparison.csv and model_comparison.csv. Approve-all cost derived as 5,223 x 1,000.')

title('09 / Performance and complexity','Compact inference, measured honestly','Timing includes preprocessing, saved model dependencies and calibration on prepared historical feature frames.')
img('latency',500)
lat=pd.read_csv(R/'latency_comparison.csv').query("model == 'LightGBM_200k'")
table(['Batch','Repeats','P50 ms','P95 ms','Transactions/s'],[[r.batch_size,r.repeats,f'{r.p50_ms:.2f}',f'{r.p95_ms:.2f}',f'{r.transactions_per_second:,.0f}'] for r in lat.itertuples()],[70,80,100,100,150])
add('LightGBM has 140 trees, a 0.113 MiB saved experiment bundle, and about 7.80 MiB measured process-RSS growth after loading and scoring. Its training/search wall time was 5.49 seconds; sampled peak process RSS was 1,031.46 MiB. Those memory measures have different scopes.','Small2')
add('<b>Boundary:</b> single process, four threads, round-robin benchmark. State retrieval, feature-history construction, HTTP and network delay are excluded. Exact benchmark hardware is not recorded in the cited comparison. These results do not establish production latency or online throughput.','Small2')
source('final_common_test/latency_comparison.csv, model_comparison.csv; models/200k/LightGBM_metadata.json; src/common_test.py.')

title('10 / Training and feature pipeline','Inside the selected LightGBM','200k names the full generated experiment; the base estimator was fitted on its 140k training rows.')
table(['Configuration','Saved value'],[['Algorithm','Gradient-boosted decision trees (gbdt)'],['Trees / leaves / learning rate','140 / 15 / 0.06'],['Regularization / min child samples','L2 (reg_lambda) = 8 / 30'],['Class imbalance','class_weight = balanced'],['Sampling','subsample = 1.0; colsample_bytree = 1.0'],['Execution','4 threads; random_state 42; deterministic; force_col_wise'],['Tuning','2 seeded trials: reg_lambda 3 vs 8, both 15 leaves'],['Winning tune-block AP','0.24661649'],['Calibration','Sigmoid, fitted on a separate 6k block']],[210,290])
sub('Train-only preprocessing')
add('Numeric columns: median imputation followed by standard scaling. Categorical columns: most-frequent imputation and dense one-hot encoding with unknown values ignored. The bundle receives 130 engineered/raw input columns; the post-encoding dimension can be larger.')
sub('Feature families')
add('Transaction and account context; amount relative to historical behavior; 5-minute to multi-day velocity windows; device and recipient novelty; cash-flow and turnaround behavior; incremental graph connectivity; location and travel; hour/channel deviation; credential and transfer sequences; agent activity. Historical state must match the offline computation at serving time.')
sub('Calibration methodology')
add('Identity, sigmoid and isotonic candidates are evaluated using separate fitting and selection blocks. Selection minimizes Brier score subject to an AP-loss limit of 0.015. The exported model uses sigmoid; probability calibration is still only validated within the simulator distribution.')
source('models/200k/LightGBM_metadata.json; src/preprocessing.py; reports/final_report.md; backend/models/final/feature_list.json.')

title('11 / Model explanations','Which inputs influence predictions?','Global SHAP attributions for the selected LightGBM from the earlier 200k experiment.')
img('shap',490)
add('Recipient familiarity is the strongest feature in this sampled explanation. Device activity and age, device risk, preferred-channel mismatch, outgoing cash flow and graph degree also contribute. This supports a multi-signal explanation, rather than a single amount threshold.')
sub('Explainability boundaries')
add('The saved analysis uses 256 randomly sampled held-out transactions for global SHAP and the 12 highest-risk examples for local explanations. These attributions describe the uncalibrated tree model, not the final calibrated probability, causal drivers, or the entire action policy. This is not a SHAP rerun on the common final test.')
sub('Evidence judges can inspect')
add('The repository includes global SHAP values, native feature importance, local explanations, fraud-type error slices and weighted rule evidence. Keep rule reasons distinguishable from learned-model attributions: a triggered rule does not prove that a particular tree caused the decision.')
source('results/200k/shap_feature_importance.csv, explainability_scope.json, local_explanations.json. Figure redrawn from saved numeric SHAP values.')

title('12 / Data scale and ablations','Does more data or complexity help?','Earlier per-scale findings are informative, but evaluate different populations and different selected systems.')
img('scales',480)
table(['Scale','Validation-selected system','Own-test AP'],[[f'{int(r.dataset_size/1000)}k',r.model,f'{r.pr_auc:.4f}'] for r in sd.itertuples()],[70,340,90])
add('The scale curve is non-monotonic. It grows population size while keeping approximate per-user history depth fixed; it is not a fixed-population learning curve and uses only one training/generation seed.','Small2')
table(['500k ablation finding','AP effect','95% interval'],[['Full hybrid minus no-anomaly fusion','+0.0043','+0.0015 to +0.0070'],['Supervised minus graph-removed retraining','+0.0018','-0.0009 to +0.0047'],['Supervised minus sequence-removed retraining','+0.0055','+0.0018 to +0.0100']],[295,75,130])
add('These earlier paired day-bootstrap intervals are conditional on fitted models (150 repeats) and not adjusted for multiple comparisons. Full hybrid was 0.0020 AP below standalone XGBoost on that own-test comparison. Correlated features can preserve proxies for removed families.','Small2')
source('results/final/dataset_size_comparison.csv; reports/final_report.md; results/final/ablation_all_sizes.csv. Not common-final-test ablations.')

title('13 / Integration status','What exists, and what remains?','The workspace now contains a Python scoring service and a configurable Next.js connection, with a rules fallback.')
table(['Repository layer','Observed implementation'],[['ML experiment','Saved models, calibrators, thresholds, dataset cards, comparisons and error analyses'],['Final exported bundle','LightGBM model, preprocessor, calibrator, feature list, metadata, policy thresholds and config snapshot'],['Supporting anomaly export','500k Isolation Forest component; separate from selected LightGBM probability'],['Next.js ML connection','src/lib/ml.ts calls the Python scoring API when MFS_ML_SERVICE_URL is configured'],['Action integration','The backend takes the more severe ML/rule action; combined high scores or blacklists can reject. Next.js preserves hard rule blocks.'],['Failure fallback','Missing service configuration or scoring failure returns DEGRADED and retains the local rule decision']],[145,355])
sub('Serving contract in the current code')
add('The Python feature service reuses offline feature engineering on a chronologically ordered replay of stored history plus the current transaction. The saved preprocessor, LightGBM and sigmoid calibrator produce probability. The decision service combines ML and rule bands; anomaly scores and SHAP factors provide supporting evidence. The Next.js client uses the returned action, with local hard blocks retained.')
sub('Remaining serving limitations')
add('The feature service replays a capped history ledger for each request, so feature work grows with history length and truncation can change long-history or lifetime graph features. Production needs incremental state, offline/online parity checks and full-request latency measurement. This report inspected integration code but did not verify a running configured service or deployment.')
add('Do not label the current app\'s 0-100 weighted rule score as the LightGBM fraud probability. The benchmark\'s model metrics do not measure the present app\'s decisions.','Small2')
source('backend/models/final/*; scripts/export_final_model.py; backend/app/services/feature_engine.py and decision_engine.py; src/lib/ml.ts, src/lib/engine.ts and src/app/api/v1/[...path]/route.ts (repository root).')

title('14 / Judge discussion','Key findings and honest limitations','Suggested presentation narrative and technical answers.')
sub('Our model choice')
add('“We exported calibrated LightGBM from the 200k experiment. On a fresh 100k synthetic comparison, it delivers 0.2417 average precision with 10.90 ms prepared-frame P95 inference. A much larger hybrid gives essentially the same ranking quality, so we prefer the simpler model.”')
sub('Our operational trade-off')
add('“At the validation-selected threshold, we flag 4.1% of transactions, with 37.0% precision and 29.1% recall. A lower reference threshold catches 70.7% of fraud but flags 33.2% of transactions. We expose this trade-off instead of claiming a single universal threshold.”')
sub('What the prototype demonstrates')
add('A reproducible synthetic benchmark; separated chronological fitting and evaluation; learned ranking compared against rules and anomaly baselines; saved calibration and threshold artifacts; uncertainty estimates; and actionable diagnostics across fraud types and workload budgets.')
sub('What it does not establish')
add('Real-world accuracy, real provider cost savings, demographic fairness, adaptive-attacker robustness, cross-generator transfer, or production serving latency. One generator family and one training seed limit generalization. The largest missed category is subtle fraud; known-device and known-recipient behavior remain difficult.')
sub('Next experiment')
add('Freeze the chosen bundle and policy, then evaluate on another untouched holdout. Repeat training seeds, include user-disjoint and temporal stress tests, estimate real costs, validate on consented de-identified event histories with delayed labels, and monitor analyst overrides and drift during shadow deployment.')
source('Synthesis of the cited local experiment artifacts. Recommendation is for a synthetic ML prototype, not autonomous live customer decisions.')

# Technical appendices preserve the complete actual input schema.
for part in range(2):
    title(f'Appendix A{part+1} / Exact schema','All 130 model inputs' if part==0 else 'Model inputs, continued','Saved preprocessor input order. IDs, target labels and post-decision outcomes are not included.')
    subset=features[part*66:(part+1)*66]
    rows=[]
    for i in range(0,len(subset),2):
        rows.append([f'{part*66+i+1:03d}  {subset[i]}',f'{part*66+i+2:03d}  {subset[i+1]}' if i+1<len(subset) else ''])
    vals=[[p(escape(a),'Feature'),p(escape(b),'Feature')] for a,b in rows]
    t=Table(vals,colWidths=[250,250]);t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('ROWBACKGROUNDS',(0,0),(-1,-1),[colors.white,colors.HexColor('#F1F6F8')]),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)]));story.append(t)
    source('backend/models/final/feature_list.json; 130 exact input names. Categorical one-hot expansion occurs inside the saved preprocessor.')

title('Appendix B / Full research grid','All models across all five scales','Average precision on each scale\'s own temporal test. Compare within columns; populations differ across columns.')
allr=pd.read_csv(ML/'results/final/all_model_results.csv')
pivot=allr.pivot_table(index='model',columns='dataset_size',values='pr_auc')
table(['System']+[f'{int(c/1000)}k' for c in pivot.columns],[[idx]+[f'{v:.4f}' for v in row] for idx,row in pivot.iterrows()],[230,54,54,54,54,54])
add('The study evaluates logistic regression, random forest, histogram gradient boosting, XGBoost, LightGBM, rules, Isolation Forest, progressive hybrid fusion, leave-one-component-out fusion and retrained feature-family removals. Hybrid fusion uses a logistic model fitted on its dedicated validation block.','Small2')
add('This table supplies the broader research comparison; it must not replace the common-test model selection table. AP is threshold-independent, while the older report\'s high-recall operating points use a different cost-selected policy.','Small2')
source('results/final/all_model_results.csv; src/train_supervised.py; reports/final_report.md.')

title('Appendix C / Rules and reproducibility','Policy details and source manifest','Report created from saved local evidence. No model was retrained and no test outcome was fabricated.')
rules=json.loads((ML/'backend/models/final/thresholds.json').read_text())['rules']
table(['Rule feature','Threshold','Weight'],[[r['feature'],r['threshold'],r['weight']] for r in rules],[330,85,85])
add('These are the 19 exported research rules. Rule-score bands are 20 / 40 / 70. Do not substitute this table for the current web app\'s independently configured rule policy.','Small2')
source('backend/models/final/thresholds.json; config.yaml. See the next page for source provenance and artifact hashes.')

title('Appendix D / Audit trail','Reproduce the evidence','Primary evidence hierarchy: latest final selection and exported metadata, common-test tables, then earlier per-scale findings.')
table(['Evidence','Local path under mfs-guard-ml/'],[['Final selection / version','results/final_common_test/final_selection.json; backend/models/final/model_metadata.json'],['Common-test scores / uncertainty','results/final_common_test/model_comparison.csv; confidence_intervals.csv'],['Policy / cost / latency','results/final_common_test/threshold_comparison.csv; business_cost_comparison.csv; latency_comparison.csv'],['Population / errors','reports/final_test_dataset_card.md; results/final_common_test/fraud_type_performance.csv; error_analysis.csv'],['Training / preprocessing','models/200k/LightGBM_metadata.json; src/train_supervised.py; src/preprocessing.py'],['Explanations / scale study','results/200k/shap_feature_importance.csv; results/final/dataset_size_comparison.csv; all_model_results.csv'],['Exports / serving boundary','scripts/export_final_model.py; backend/app/services/; backend/models/final/; repository-root src/lib/ml.ts and src/lib/engine.ts']],[155,345])
sub('SHA-256 of the selected exported components')
for filename in ['model.joblib','preprocessor.joblib','calibrator.joblib','feature_list.json','thresholds.json']:
    digest=hashlib.sha256((ML/'backend/models/final'/filename).read_bytes()).hexdigest()
    add(f'<b>{filename}</b><br/>{digest[:32]}<br/>{digest[32:]}','Small2')
add('Figures were redrawn from saved numeric artifacts; the architecture image is a technical illustration. The report builder is tmp/pdfs/build_report.py at the repository root. Use the saved joblib bundles only in a compatible Python environment; preserve the feature list, preprocessing, calibration and versioned policy together.','Small2')

def footer(canvas,doc):
    canvas.setStrokeColor(colors.HexColor('#CDDDE4'));canvas.line(46,43,549,43)
    canvas.setFont('Body',8);canvas.setFillColor(colors.HexColor(GREY));canvas.drawString(46,30,'MFS GUARD  |  Synthetic model findings  |  07 Oct 2026');canvas.drawRightString(549,30,f'{doc.page:02d}')
    canvas.setFillColor(colors.HexColor(TEAL));canvas.rect(0,822,595,20,fill=1,stroke=0)

pdf=OUT/'MFS_Guard_Model_Findings_Judges.pdf'
doc=SimpleDocTemplate(str(pdf),pagesize=(595.28,841.89),rightMargin=46,leftMargin=46,topMargin=47,bottomMargin=56,title='MFS Guard - Model Findings for Judges',author='MFS Guard',subject='Synthetic fraud model comparisons, selected LightGBM, technical evidence and limitations')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
from pypdf import PdfReader
reader=PdfReader(pdf)
for i,page in enumerate(reader.pages):
    text=page.extract_text()
    print(f'Page {i+1:02d}: {len(text)} characters | {text[:95].replace(chr(10), " / ")}')
assert len(features)==130
assert chosen.true_positive+chosen.false_positive+chosen.true_negative+chosen.false_negative==100000
assert chosen.true_positive+chosen.false_negative==5223
print(f'Created {pdf} ({len(reader.pages)} pages)')
