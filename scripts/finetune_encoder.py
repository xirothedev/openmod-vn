#!/usr/bin/env python3
"""Bake-off finetune for L3 fast encoder (3-way + span BIO head, OOM-safe).

Same data/config for every base model; winner picked by eval MF1 + ONNX p99 gate.
Runs on bvpt3 without stopping services (micro-batch 8, fp16, <10GB VRAM).

Usage (on bvpt3):
  ~/venvs/openmod-train/bin/python scripts/finetune_encoder.py \
    --model Fsoft-AIC/videberta-xsmall --data data/l3_vi.jsonl --out ~/openmod-train/bakeoff/videberta
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

LABELS = ["ALLOW", "BORDERLINE", "BLOCK"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
CLS_LABELS = LABELS
BIO_LABELS = ["O", "B", "I"]


def load_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def strat_split(rows: list[dict], frac: float, seed: int):
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["label"], []).append(r)
    rnd = random.Random(seed)
    train, ev = [], []
    for rs in by.values():
        rnd.shuffle(rs)
        n = max(1, int(len(rs) * frac))
        ev += rs[:n]
        train += rs[n:]
    rnd.shuffle(train)
    rnd.shuffle(ev)
    return train, ev


def encode(tok, row: dict, seq_max: int):
    text = f"{row['phrase']} [SEP] {row['left']} [SEP] {row['right']}"
    enc = tok(
        text,
        max_length=seq_max,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
    )
    tags = [0] * seq_max
    for span in row.get("spans") or []:
        needle = span.strip()
        if not needle:
            continue
        start = text.find(needle)
        if start < 0:
            continue
        end = start + len(needle)
        first = True
        for i, (s, e) in enumerate(enc["offset_mapping"]):
            if e <= start or s >= end or s == e == 0:
                continue
            tags[i] = 1 if first else 2
            first = False
    return enc["input_ids"], enc["attention_mask"], tags


def macro_f1(gold: list[int], pred: list[int], n: int = 3) -> tuple[float, list[float]]:
    f1s = []
    for c in range(n):
        tp = sum(1 for g, p in zip(gold, pred) if g == p == c)
        fp = sum(1 for g, p in zip(gold, pred) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold, pred) if g == c and p != c)
        if tp == 0:
            f1s.append(0.0)
        else:
            prec, rec = tp / (tp + fp), tp / (tp + fn)
            f1s.append(2 * prec * rec / (prec + rec))
    return sum(f1s) / n, f1s


def main() -> int:
    ap = argparse.ArgumentParser(description="L3 encoder bake-off finetune")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--seq-max", type=int, default=96)
    ap.add_argument("--eval-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--export-only", action="store_true")
    a = ap.parse_args()

    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = load_rows(Path(a.data))
    train_rows, eval_rows = strat_split(rows, a.eval_frac, a.seed)
    print(
        f"bakeoff: model={a.model} train={len(train_rows)} eval={len(eval_rows)} device={device}"
    )

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=False)
    base = AutoModel.from_pretrained(a.model, trust_remote_code=False)
    hidden = base.config.hidden_size

    class Judge(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = base
            self.drop = nn.Dropout(0.1)
            self.cls = nn.Linear(hidden, 3)
            self.bio = nn.Linear(hidden, 3)

        def forward(self, ids, mask):
            h = self.drop(self.base(ids, mask).last_hidden_state)
            return self.cls(h[:, 0]), self.bio(h)

    net = Judge().to(device)
    dist = Counter(r["label"] for r in train_rows)
    cw = torch.tensor(
        [len(train_rows) / dist[l] for l in LABELS], dtype=torch.float32
    ).to(device)
    ce = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    def batchify(rs, shuffle):
        idx = list(range(len(rs)))
        if shuffle:
            random.Random(a.seed).shuffle(idx)
        for i in range(0, len(idx), a.micro_batch):
            chunk = [rs[j] for j in idx[i : i + a.micro_batch]]
            ids, masks, tags, labs = [], [], [], []
            for r in chunk:
                ii, mm, tt = encode(tok, r, a.seq_max)
                ids.append(ii)
                masks.append(mm)
                tags.append(tt)
                labs.append(LABEL2ID[r["label"]])
            yield (
                torch.tensor(ids).to(device),
                torch.tensor(masks).to(device),
                torch.tensor(tags).to(device),
                torch.tensor(labs).to(device),
            )

    @torch.no_grad()
    def evaluate():
        net.eval()
        gold, pred = [], []
        for ids, mask, _, labs in batchify(eval_rows, False):
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits, _ = net(ids, mask)
            gold += labs.tolist()
            pred += logits.argmax(-1).tolist()
        mf1, f1s = macro_f1(gold, pred)
        acc = sum(g == p for g, p in zip(gold, pred)) / len(gold)
        return mf1, f1s, acc

    best, best_state = -1.0, None
    step = 0
    if a.export_only:
        best_state = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
        net.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print("bakeoff: export-only, loaded best.pt")
    else:
        for ep in range(a.epochs):
            net.train()
            for bi, (ids, mask, tags, labs) in enumerate(batchify(train_rows, True)):
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    logits, bio = net(ids, mask)
                    loss = (
                        ce(logits, labs) + ce(bio.view(-1, 3), tags.view(-1))
                    ) / a.grad_accum
                scaler.scale(loss).backward()
                if (bi + 1) % a.grad_accum == 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad()
                    step += 1
            mf1, f1s, acc = evaluate()
            print(
                f"bakeoff: epoch={ep + 1} mf1={mf1:.4f} f1={f1s} acc={acc:.4f}",
                flush=True,
            )
            if mf1 > best:
                best = mf1
                best_state = {k: v.cpu() for k, v in net.state_dict().items()}

        torch.save(best_state, out / "best.pt")
        net.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    mf1, f1s, acc = evaluate()
    metrics = {
        "model": a.model,
        "mf1": mf1,
        "f1_per_class": dict(zip(CLS_LABELS, f1s)),
        "acc": acc,
        "n_train": len(train_rows),
        "n_eval": len(eval_rows),
        "eff_batch": a.micro_batch * a.grad_accum,
    }
    print(f"bakeoff: BEST mf1={mf1:.4f} acc={acc:.4f}")

    if not a.skip_export:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            net.load_state_dict({k: v.to(device) for k, v in best_state.items()})
            net.eval()
            dummy = {
                "ids": torch.ones(1, a.seq_max, dtype=torch.long).to(device),
                "mask": torch.ones(1, a.seq_max, dtype=torch.long).to(device),
            }

            class ClsOnly(nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m

                def forward(self, ids, mask):
                    return self.m(ids, mask)[0]

            onnx_path = out / "model.onnx"
            # Fixed batch1 shapes (gate measures batch1 seq96): friendlier to
            # shape inference than dynamic axes.
            torch.onnx.export(
                ClsOnly(net),
                (dummy["ids"], dummy["mask"]),
                str(onnx_path),
                input_names=["ids", "mask"],
                output_names=["logits"],
                opset_version=17,
            )
            quantize_dynamic(
                str(onnx_path),
                str(out / "model.int8.onnx"),
                weight_type=QuantType.QInt8,
            )

            import numpy as np
            import onnxruntime as ort

            sess = ort.InferenceSession(
                str(out / "model.int8.onnx"), providers=["CPUExecutionProvider"]
            )
            feed = {
                "ids": np.ones((1, a.seq_max), dtype=np.int64),
                "mask": np.ones((1, a.seq_max), dtype=np.int64),
            }
            for _ in range(10):
                sess.run(None, feed)
            ts = []
            for _ in range(200):
                t0 = time.perf_counter()
                sess.run(None, feed)
                ts.append((time.perf_counter() - t0) * 1000)
            ts.sort()
            metrics["p99_ms"] = ts[int(len(ts) * 0.99)]
            print(f"bakeoff: p99={metrics['p99_ms']:.2f}ms CPU batch1 seq{a.seq_max}")
        except Exception as e:  # noqa: BLE001 - export bench is best-effort; train result kept
            print(f"bakeoff: export SKIP ({type(e).__name__}: {str(e)[:120]})")

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
