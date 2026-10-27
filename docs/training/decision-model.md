# Decision model training

## Pick

- Fast path (sync, <15ms CPU): `Fsoft-AIC/videberta-xsmall` (22M DeBERTaV3, raw text, MIT-lineage repo — verify HF weights license before prod). Fallback: `FacebookAI/xlm-roberta-base` (MIT).
- Deep judge (async GPU): `Qwen/Qwen2.5-3B-Instruct` (Apache-2.0, native JSON). Approved swap: `Qwen3-4B` (same license family, better VI).
- Teachers (offline, research-only): ViSoBERT / ViHateT5 for labeling + distillation. Never ship their weights.

## Data

`l3_vi.jsonl` rows: `phrase, left, right, label (BLOCK|BORDERLINE|ALLOW), spans, policy, is_teencode`.
Pair encoding: `phrase [SEP] left [SEP] right`, seq <= 96. Stratify HATE/OFF/CLEAN + teencode splits. Mine hard negatives from judge disagreements.

## Steps

1. Teacher-label ViHOS spans + HateCheck-VI templates with ViHateT5/ViSoBERT.
2. Human review every teacher label before it enters `data/`.
3. Fine-tune videberta-xsmall 3-way + span BIO head: AdamW 2e-5, 10 epochs, batch 32, distill T=2.
4. Export ONNX opset17, dynamic int8 quant, gate p99 <15ms batch1 seq96 CPU.
5. LoRA r=16 Qwen judge on ViHSD+ViCTSD+ViHOS-as-instruct (~40k rows), vLLM `guided_json`, temp 0, max 256 tokens.
6. Nightly distill judge corrections back into encoder set. Weekly refresh.

## Eval

HateCheck-VI functional suite + hard negatives + bias slices (religion, gender, region). Report MF1 + schema-valid % + p99 latency.
