# Filter classifier v1 — train plan (binary CLEAN|TOXIC, full-text)

Agreed after grill Q13-Q16. Prep only, no training run yet.

## 1. Label + input

- Binary `CLEAN|TOXIC` + keep `severity 0-4` for policy mapping
  (`kid_safe/standard_social/relaxed_gaming` thresholds unchanged).
- Map from L3: `ALLOW->CLEAN`, `BORDERLINE|BLOCK->TOXIC`
  (`scripts/build_classifier_data.py`). Benchmarks ViHOS map
  `CLEAN->CLEAN`, `OFFENSIVE|HATE->TOXIC`.
- Input: **full text** (chat message), not `phrase [SEP] left [SEP] right`.
  Fragment rows in `l3_vi.jsonl` convert via `text` field.
- Schema: `data/schemas/classifier_sample.schema.json`.

## 2. Data

- Build: `python3 scripts/build_classifier_data.py --out data/classifier_vi.jsonl`
  (l3 9918 + bench 4829, dedup by id, stratify CLEAN/TOXIC + teencode).
- Hard negatives next: shadow diffs (encoder FP on teacher-ALLOW, FN on
  teacher-BLOCK) from `decision-v1-bvpt3-plan.md` §8. Not in v1 seed.
- Human review stays: teacher labels never enter `data/` unreviewed.
- Eval: HateCheck-VI templates + hard negatives + bias slices
  (religion/gender/region from `finetune_encoder.py:26-38`).

## 3. Model + export

- Base: `FacebookAI/xlm-roberta-base` (MIT, bake-off winner MF1 0.823).
  `videberta-xsmall` stays eval-only (no weights license).
- Head: CLS 2-way only (drop BIO span head for v1 filter).
  Reuse OOM-safe settings: micro-batch 8, accum 4 (=32), fp16, workers 2,
  seq 96, AdamW 2e-5, 10 epochs, focal + balanced.
- Export: ONNX opset17 dynamic int8, gate p99 <15ms CPU batch1 seq96.
  `model.int8.onnx` + `tokenizer.json` + `metrics.json` + `threshold.json`
  (flag tau fitted on ViHSD half/half, live point T=1.5 tau=0.5).
- Web export deferred: `onnxruntime-web` / `transformers.js` after Python
  gate passes. Examples call `POST /moderate` until then.

## 4. Relation to current pipeline

- ViGuard-X today is L3 3-way+BIO judge in shadow mode, heuristic decides.
  Classifier v1 does **not** replace it on day one: ship as `shadow` sidecar,
  compare binary vs heuristic 5-way, promote to `veto/flag` only after
  clean-eval precision gate (no point is auto-block safe at F1 0.33 today).
- Future single-model (jev-like): direct text->Decision 5-way. Needs diverse
  data + reviewed hard negatives + multi-label heads (v3 notes in plan §9).

## 5. Steps

1. `scripts/build_classifier_data.py` -> `data/classifier_vi.jsonl` (reviewed).
2. `scripts/train_classifier.py --check/--smoke` on bvpt3 (1 job, tmux,
   kill-switch RAM<8G/VRAM<4G), then full finetune.
3. Export ONNX + fit threshold (`scripts/fit_thresholds.py` pattern).
4. Release `v0.3.0`: weights + `threshold.json`, CD attach (fix: include
   `viguard-x/` + `*.json` in `cd.yml`, today missing).
5. Examples switch `REMOTE_API` to classifier-backed `/moderate` with no
   client change (Decision shape unchanged).
