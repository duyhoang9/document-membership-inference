"""Create artificial token-loss data ONLY to smoke-test the evaluation pipeline."""
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
rows = []
for doc in range(100):
    member = int(doc < 50)
    # Deliberately artificial class separation; NOT real model scores.
    losses = np.clip(rng.normal(2.0 if member else 2.35, 0.85, 200), 0.01, None)
    for i, loss in enumerate(losses):
        rows.append((f"demo_{doc:03d}", member, i, float(loss)))
out = Path('data/demo/token_losses.csv')
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows, columns=['document_id', 'member', 'token_index', 'loss']).to_csv(out, index=False)
print(f'Wrote synthetic demonstration data: {out}')
