# Document-Level Membership Inference — CSI 5100

Compare mean token loss with histogram features and a random forest. Inspired by Meeus et al., *Did the Neurons Read Your Book?* (arXiv:2310.15007). This is a simplified experiment, not an exact reproduction.

## Windows PowerShell (Python 3.11 or 3.12)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/run_experiment.py --per-class 100 --epochs 3
.\.venv\Scripts\python.exe src/evaluate.py --input data/real/before.csv --output results/before
.\.venv\Scripts\python.exe src/evaluate.py --input data/real/token_losses.csv --output results/after
```

First run downloads AG News and DistilGPT2 from Hugging Face. CPU is supported but training may be slow. For a smoke test use `--per-class 20 --epochs 1` (not final evidence). No API key needed.

## Design
Randomly assign distinct 64-token excerpts from the same news dataset to fine-tuning members and held-out nonmembers. Score both groups before and after fine-tuning. Fit histogram bin boundaries and random forest on attack-training documents only, then evaluate on a disjoint stratified test set. The same seed and document IDs give the same attack split before and after. Membership means inclusion in our fine-tuning run, not original pretraining. Exact duplicate excerpts are removed; near duplicates may remain. Short excerpts are not full books. No original-paper normalization is implemented. Report observed results even if the histogram method loses. Synthetic demo is only a software test.

## Outputs
`data/real/config.json`, `before.csv`, `token_losses.csv`; `results/before` and `results/after` contain metrics, ROC curve and random forest confusion matrix. No real results are bundled; run locally to obtain them.
