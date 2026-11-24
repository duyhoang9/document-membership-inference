"""Controlled fine-tuning membership experiment on distinct AG News articles."""
import argparse, csv, json, random
from pathlib import Path
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--per-class', type=int, default=100)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--model', default='distilgpt2')
    a = p.parse_args()
    if a.per_class < 10 or a.epochs < 1:
        p.error('Use at least 10 documents per class and one epoch')
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model).to(device)
    ds = load_dataset('ag_news', split='train').shuffle(seed=a.seed)
    # Same source, equal lengths, random assignment; remove identical token excerpts.
    docs, seen = [], set()
    for row in ds:
        ids = tuple(tok.encode(row['text'], add_special_tokens=False, truncation=True, max_length=64))
        if len(ids) != 64 or ids in seen:
            continue
        seen.add(ids)
        docs.append(ids)
        if len(docs) == 2*a.per_class:
            break
    if len(docs) != 2*a.per_class:
        raise RuntimeError('Not enough unique excerpts')
    random.shuffle(docs)
    labels = [1]*a.per_class + [0]*a.per_class
    out = Path('data/real'); out.mkdir(parents=True, exist_ok=True)
    def score(path):
        model.eval()
        with path.open('w', newline='', encoding='utf-8') as f, torch.no_grad():
            w = csv.writer(f); w.writerow(['document_id','member','token_index','loss'])
            for i, ids in enumerate(docs):
                x = torch.tensor([ids], device=device)
                logits = model(x).logits[:, :-1, :]
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x[:,1:].reshape(-1), reduction='none')
                w.writerows((i, labels[i], j+1, v) for j,v in enumerate(loss.cpu().tolist()))
    print('Device:', device, '| Scoring before fine-tuning', flush=True)
    score(out/'before.csv')
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
    for epoch in range(a.epochs):
        model.train(); order = list(range(a.per_class)); random.shuffle(order); total = 0
        for i in range(0,len(order),4):
            x = torch.tensor([docs[j] for j in order[i:i+4]], device=device)
            opt.zero_grad(); loss = model(x, labels=x).loss; loss.backward(); opt.step()
            total += loss.item()*len(x)
        print(f'Epoch {epoch+1}: training loss {total/a.per_class:.4f}', flush=True)
    score(out/'token_losses.csv')
    (out/'config.json').write_text(json.dumps(vars(a)|{'device':str(device),'tokens_per_excerpt':64,'dataset':'ag_news','membership':'controlled fine-tuning only'}, indent=2))
    print('Saved before and after token losses. Run evaluate.py next.')

if __name__ == '__main__':
    main()
