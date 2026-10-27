# Decision v1 — train plan trên bvpt3 (chống OOM)

Thống nhất sau grill 4 vòng. Chưa train, chỉ plan.

## 1. Chốt scope

- V1 = cả 2 tầng, chạy nối tiếp, không song song:
  - Fast encoder sync: `Fsoft-AIC/videberta-xsmall` (22M), fallback `FacebookAI/xlm-roberta-base` (MIT).
    Lý do giữ encoder: sync `POST /moderate` cần p99 <15ms CPU batch1 seq96.
    Qwen 0.6B đã nặng gấp ~30x, Qwen 4B cần GPU giây/lượt → chỉ làm judge async.
  - Deep judge async: smoke `Qwen/Qwen3-0.6B` rồi chính `Qwen/Qwen3-4B` (Apache-2.0, verify weights trước prod).
    Bỏ `Qwen2.5-3B` ở v1 để khỏi phân tán. QLoRA r=16 4-bit, không full-finetune.
- Label: encoder học 3-way `BLOCK|BORDERLINE|ALLOW` + span BIO, rồi heuristic map sang Decision 5-way
  (`ALLOW|MASK|WARN|BLOCK|ESCALATE`) + severity 0-4 theo policy. Không học thẳng 5-way ở v1.
- Data: smoke-test pipeline trên 4865 benchmark rows + 13 rules hiện có trước,
  song song teacher-label offline + human review 100% để tạo `data/l3_vi.jsonl`. Không train trên label chưa review.

## 2. Facts bvpt3 (đã đo ngày 27-10-2026)

- Host `ai-inf-03`: RAM 121Gi (còn khả dụng ~48Gi), GPU `NVIDIA GB10` 124610MB unified, disk còn 3.1T.
- Đang chạy: VLLM reranker 8B ~21GB + Ollama ~5GB + Whisper ~3GB + RagFlow/MySQL/Grafana.
  User cho phép dừng tạm vì không ai dùng → phải dừng graceful, có approve, có rollback, chỉ dừng trong cửa sổ train.
- Python 3.12, đã có `torch 2.13 + cu130`, `transformers 5.13`, `datasets`, `accelerate`, `onnxruntime`.
  Thiếu: `peft`, `bitsandbytes`, `optimum[onnxruntime]`, `vllm` client. Cài vào venv riêng, không đụng system pip.
- Repo chưa có `data/l3_vi.jsonl`. Staging thô: `vihos_train.csv` 8854 + `hatecheck_cases.csv` 3728.
  Skeleton `scripts/train_decision_model.py --check` READY (2+2 contrastive đủ).

## 3. Ngưỡng chống OOM (cứng)

- Chỉ 1 job train tại 1 thời điểm, chạy trong `tmux`, log VRAM/RAM mỗi 30s.
- Dừng train ngay (kill-switch) khi RAM khả dụng <8GB hoặc VRAM trống <4GB hoặc p99 host tăng đột biến.
- Không chạy vLLM serve song song khi train judge. Export ONNX chạy trên CPU.
- Encoder: micro-batch 8, accum 4 (=32 effective), fp16, workers 2, seq ≤96, grad-clip 1.0, checkpoint mỗi 500 steps.
- Judge QLoRA: micro-batch 1–2, accum 8–16, seq in ≤512 / out ≤256, temp 0, grad-checkpoint on, 4-bit NF4, attn FP16.
- HF cache + output dir đặt trên `/` (còn 3.1T), không đặt `/tmp` RAM-disk.

## 4. Steps 0–6

### Step 0 — Prep + license + snapshot (trên bvpt3)

```bash
ssh bvpt3
nvidia-smi; free -h
huggingface-cli download Fsoft-AIC/videberta-xsmall --dry-run 2>&1 | head
python3 -c "from huggingface_hub import model_info; print(model_info('Qwen/Qwen3-4B').cardData)"
# snapshot service đang chạy
systemctl --user list-units 2>/dev/null | head; ps aux --sort=-%mem | head
# dừng graceful theo approve, ghi lại lệnh bật lại vào ROLLBACK.txt
python3 -m venv ~/venvs/openmod-train && ~/venvs/openmod-train/bin/pip install -U "peft>=0.12" "bitsandbytes>=0.43" "optimum[onnxruntime]>=1.20" "huggingface_hub>=0.24"
```

Gate 0: license videberta + Qwen3-0.6B/4B đã ghi vào plan, ROLLBACK.txt có lệnh bật lại service.

### Step 1 — Smoke-test (không cần dừng service)

```bash
ssh bvpt3 'cd ~/openmod-vn && ~/venvs/openmod-train/bin/python scripts/train_decision_model.py --check'
# thử load tokenizer + 1 batch encoder micro-8 trên GPU lát mỏng
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ~/venvs/openmod-train/bin/python -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('Fsoft-AIC/videberta-xsmall'); print(t('chăn rau [SEP] test [SEP] test')}"
```

### Step 2 — Teacher-label offline + human review (chặn train nếu chưa xong)

- Chạy ViHateT5/ViSoBERT offline trên `data/_staging/` + ViHOS spans + HateCheck-VI templates.
- Mọi label phải qua human review trước khi vào `data/` (validator bắt `source/source_url/license/retrieved_at`).
- Nguồn research-only (ViHSD, ViCTSD, VOZ-HSD) eval-only, không bundle. ViLexNorm NC chỉ preprocess.
- Target v1: `data/l3_vi.jsonl` với `phrase, left, right, label, spans, policy, is_teencode`, pair `phrase [SEP] left [SEP] right`.

### Step 3 — Finetune encoder (dừng service, chạy 1 mình)

AdamW 2e-5, 10 epochs, eff-batch 32, distill T=2, 3-way + span BIO head. Seq ≤96, stratify HATE/OFF/CLEAN + teencode.
Xong giải phóng VRAM hoàn toàn trước Step 5.

### Step 4 — Export ONNX + gate

ONNX opset17, dynamic int8 quant. Gate: p99 <15ms CPU batch1 seq96, schema-valid %, MF1 trên HateCheck-VI + hard negatives + bias slices (religion, gender, region). Rớt gate → không thay `HeuristicContextJudge`.

### Step 5 — Judge QLoRA (dừng service, chạy 1 mình)

- Smoke `Qwen/Qwen3-0.6B` QLoRA r=16 trước để kiểm tra OOM + `guided_json` schema.
- Đạt mới train `Qwen/Qwen3-4B` QLoRA r=16 trên ViHSD+ViCTSD+ViHOS-as-instruct (~40k rows).
- Serve thử bằng vLLM `guided_json`, temp 0, max 256 tokens, đo schema-valid %.

### Step 6 — Distill (defer sang nightly sau v1)

Judge corrections → encoder set, refresh weekly. V1 chỉ cần đường thu thập log, chưa cần cron.

## 6. Step 0–1 results (2026-10-27, bvpt3)

- Snapshot: `~/openmod-train/ROLLBACK.txt`. RAM 48Gi khả dụng, GPU ~30GB đang dùng. Không dừng service.
- Venv `~/venvs/openmod-train`: peft 0.21.2, bitsandbytes 0.50.2, optimum 2.1.0, torch 2.14.1, transformers 4.57.6.
- License gate (`~/openmod-train/LICENSE-GATE.txt`): Qwen3-0.6B/4B Apache-2.0 OK;
  xlm-roberta-base MIT OK → prod fallback v1; videberta-xsmall không license → eval-only.
- `--check`: READY (13 rules, 4865 rows). `--smoke`: PASS, RAM không đổi.
- Tokenizer trial: xlm-roberta vocab 250002 OK, videberta vocab 128000 OK.

## 7. Bake-off + clean-bench results (2026-10-27/28, bvpt3)

Encoder v2 (focal γ=2 + balanced, dual-head CLS+BIO, ONNX dynamo + int8):

| model | MF1 (976 eval) | BORDERLINE F1 | int8 p99 CPU b1/s96 |
| --- | --- | --- | --- |
| xlm-roberta-base | 0.823 | 0.733 | 24.9ms |
| videberta-xsmall | 0.781 | 0.641 | 19.4ms |

Benchmarks 4865 rows overlap train 96.6% → recall numbers inflated, ranking only:
xlm-r τ=0.5 (FP 13, FN 378) > videberta τ=0.5 (FP 15, FN 917).

Clean eval, ViHSD test 6680 (overlap 18%, eval-only permitted): heuristic P 0.986/R 0.115.
Best hybrid F1 = xlm-r τ=0.7 (P 0.227, R 0.595, F1 0.33). No point is auto-block safe.
Decision: ship encoder as **shadow sidecar** (default `shadow=True`, rationale-annotated,
heuristic still decides); collect diffs → hard negatives → v2. Gate 15ms stands, unmet.

## 8. v2 notes (2026-10-28)

- License videberta: repo `Fsoft-AIC/ViDeBERTa` LICENSE = MIT (code), but HF weights
  card declares nothing. Options: (a) prod-use with attribution + note, (b) ask Fsoft-AIC
  for a license tag. Decision pending with user.
- Shadow mining (xlm-r, 9918 train texts): encoder FP on 4158 teacher-ALLOW rows
  (over-flags clean text), FN on 72 teacher-BLOCK rows. Fix path: temperature scaling
  (added `temperature` to judge + pipeline) fit on held-out eval, not retrain alone.
- v2 retrain: xlm-r on 9918 rows (S5 included), then fit T + re-bench clean.

## 9. v2 + calibration results (2026-10-28)

- v2 xlm-r (9918 rows): MF1 0.8214, int8 p99 15.84ms CPU b1/s96 (near gate).
- Threshold fit (`scripts/fit_thresholds.py`, ViHSD half/half): best fit T=1.5 τ=0.5
  F1 0.349 → report half F1 0.324 (P 0.233, R 0.531). T=1.0 τ=0.5 was F1 0.332.
- Verdict: temperature helps modestly; no point is auto-block safe (P ≤ 0.38 everywhere
  with R > 0.25). Shadow sidecar stands. Fitted live point on record: T=1.5, τ=0.5.
- Next levers: diverse data + reviewed hard negatives + multi-label heads (v3).

## 5. Done criteria v1

- `data/l3_vi.jsonl` có review, validator xanh, 17 tests + ruff sạch.
- Encoder ONNX int8 qua gate p99 + MF1 + bias slices.
- Judge QLoRA 4B schema-valid, không OOM, ROLLBACK service thành công, bvpt3 trả về trạng thái ban đầu.
- `code-review` (Standards + Spec) trước merge, `research` verify license weights đã ghi nhận.
