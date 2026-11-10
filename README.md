# Document-Level Membership Inference (CSI 5100)

A small-scale, **planned** reproduction inspired by *Did the Neurons Read Your Book? Document-level Membership Inference for Large Language Models*.

## Research question
Can a document's token-loss distribution distinguish documents used for model training from held-out documents better than average token loss alone?

## Current status
- [x] Working **synthetic-data pipeline** for testing feature extraction and evaluation code.
- [ ] Select documents and establish **verified membership labels**.
- [ ] Train or fine-tune a small language model on member documents only.
- [ ] Extract actual token-level losses for each document.
- [ ] Run real evaluation and analyze dataset shift / document length confounders.

**Do not interpret demo metrics as real membership-inference results.**

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python src/make_demo_data.py
python src/evaluate.py --input data/demo/token_losses.csv
```

The evaluation prints test-set ROC-AUC and accuracy and writes `results/metrics.csv`, `results/roc_curve.png`, and `results/confusion_matrix.png`.

## Input format
A CSV with columns `document_id`, `member`, `token_index`, `loss`. Each document has many rows, one for each token. `member` is 1 if that document was used to train the target model, otherwise 0. **Membership must be established independently** (not guessed based on document age).

## Evaluation design
Documents, not tokens, are split into training and test sets. Histogram features are fitted on training documents only. Two methods are compared: mean token loss (lower predicts membership) and a histogram of losses with random-forest classification. The synthetic demo uses an artificial signal and is strictly a software smoke test.

## Proposed real experiment
Fine-tune a small open-weight causal language model on a controlled subset of public-domain documents. Keep other documents completely held out. Extract next-token negative log likelihood for all documents, processing long texts in overlapping or non-overlapping chunks with correct masking at boundaries. Split document IDs into disjoint sets for fitting the attack and final evaluation; avoid overlapping excerpts and duplicates. Match lengths, sources, and genres where possible.

## Notes
GitHub is for versioning code, not for querying ChatGPT. The ChatGPT Plus website does not provide token-level losses; use an open-weight model with local access to logits for the real experiment.
