"""Evaluate document-level membership from token losses; demo input is synthetic."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='data/demo/token_losses.csv')
parser.add_argument('--output', default='results')
args = parser.parse_args()
output = Path(args.output)
df = pd.read_csv(args.input)
required = {'document_id', 'member', 'loss'}
if not required.issubset(df.columns):
    raise ValueError(f'Missing columns: {required - set(df.columns)}')
if df[['document_id', 'member', 'loss']].isna().any().any():
    raise ValueError('Null document IDs, labels, or losses are not allowed')
if not set(df.member.unique()).issubset({0, 1}):
    raise ValueError('member must be binary 0/1')
if (df.groupby('document_id').member.nunique() != 1).any():
    raise ValueError('Each document must have one membership label')
groups = list(df.groupby('document_id', sort=True))
ids = [doc for doc, _ in groups]
y = np.array([int(g.member.iloc[0]) for _, g in groups])
losses = [g.loss.to_numpy(dtype=float) for _, g in groups]
if len(ids) < 12 or min(np.bincount(y, minlength=2)) < 6:
    raise ValueError('Need at least 12 documents and 6 of each class')
train_idx, test_idx = train_test_split(np.arange(len(ids)), test_size=.3, random_state=42, stratify=y)
# Histogram edges derived from TRAINING documents only, avoiding test-data leakage.
train_losses = np.concatenate([losses[i] for i in train_idx])
edges = np.quantile(train_losses, np.linspace(0, 1, 11))
edges[0], edges[-1] = -np.inf, np.inf
# Repeated quantiles may occur in unusual data; make edges strictly increasing.
edges = np.unique(edges)
if len(edges) < 3:
    raise ValueError('Loss values have insufficient variation for histogram features')
X = np.array([np.histogram(x, bins=edges)[0] / len(x) for x in losses])
mean_loss = np.array([x.mean() for x in losses])
rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=42)
rf.fit(X[train_idx], y[train_idx])
rf_score = rf.predict_proba(X[test_idx])[:, 1]
# Lower loss means more likely a member. This baseline is evaluated directly, no fit needed.
base_score = -mean_loss[test_idx]
y_test = y[test_idx]
metrics = []
output.mkdir(parents=True, exist_ok=True)
roc_fig, roc_ax = plt.subplots(figsize=(7, 5))
for name, score, threshold in [('Mean token loss', base_score, None), ('Histogram + random forest', rf_score, .5)]:
    auc = roc_auc_score(y_test, score)
    if threshold is None:
        # Threshold selected from TRAINING data only (median of training means).
        pred = (mean_loss[test_idx] < np.median(mean_loss[train_idx])).astype(int)
    else:
        pred = (score >= threshold).astype(int)
    acc = accuracy_score(y_test, pred)
    metrics.append({'method': name, 'roc_auc': round(auc, 4), 'accuracy': round(acc, 4), 'test_documents': len(test_idx)})
    fpr, tpr, _ = roc_curve(y_test, score)
    roc_ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
    if name == 'Histogram + random forest':
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, colorbar=False)
        ax.set_title('Histogram + random forest')
        fig.tight_layout()
        fig.savefig(output / 'confusion_matrix.png', dpi=150)
        plt.close(fig)
roc_ax.plot([0, 1], [0, 1], 'k--', label='Random guess')
roc_ax.set_xlabel('False positive rate')
roc_ax.set_ylabel('True positive rate')
roc_ax.set_title('Document-level membership inference')
roc_ax.legend()
roc_fig.tight_layout()
roc_fig.savefig(output / 'roc_curve.png', dpi=150)
pd.DataFrame(metrics).to_csv(output / 'metrics.csv', index=False)
print(pd.DataFrame(metrics).to_string(index=False))
print(f'Saved metrics and plots in {output}')
if 'demo' in args.input:
    print('WARNING: These are SYNTHETIC demo metrics, not research findings.')
