# Decision/Judge Models: how production systems train them today

Context for OUR v1: sync encoder gate p99 <15ms CPU batch1 seq96 via ONNX int8;
async Qwen3-4B QLoRA judge; judge->encoder distillation planned.
Measured: xlm-roberta-base MF1 0.824 but p99-int8 51ms (misses gate);
videberta-xsmall MF1 0.774, p99 14.7ms (passes). All sources below verified live.

## 1. Encoder toxicity classifiers

- **Jigsaw/Perspective**: production API scores TOXICITY as fraction of raters
(flag 0.6 = 6/10 raters) ([model cards](https://developers.perspectiveapi.com/s/about-the-api-model-cards?language=en_US)).
Latest generation is a multilingual **character-level Transformer** ("Unified Toxic
Content Classification", [ACM 10.1145/3534678.3539147](https://dl.acm.org/doi/10.1145/3534678.3539147)) —
char-level handles obfuscation (l33t, diacritics) that kills subword encoders.
- **Detoxify** ([github.com/unitaryai/detoxify](https://github.com/unitaryai/detoxify)):
multi-head BERT-family finetunes on 3 Jigsaw challenges. `original`=bert-base (AUC 98.64),
`unbiased`=roberta-base (AUC 93.74, trained on combined 2018+2019 data),
`multilingual`=xlm-roberta-base (AUC 92.11). Label design is **multi-label heads**
(toxic, severe, obscene, threat, insult, identity_attack, sexual_explicit).
Bias-key move: 2019 challenge adds **identity-mention auxiliary labels**
(male/female/gay/christian/jewish/muslim/black/white/...) and a bias-aware AUC metric
([Kaggle eval](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/overview/evaluation)).
- **HateBERT** ([arXiv:2010.12472](https://arxiv.org/abs/2010.12472)): continued-pretrains
BERT on RAL-E (Reddit comments from banned abusive communities), then finetunes.
Beats generic BERT on all 3 EN abuse datasets tested — i.e. **domain-adaptive pretraining
on in-domain toxic text** beats bigger generic models. Direct precedent for our
VOZ-style pretraining idea.
- **ToxiGen** ([arXiv:2203.09509](https://arxiv.org/abs/2203.09509)): 274k GPT-3-generated
implicit/adversarial toxic+benign statements over 13 minority groups; 94.5% of toxic
examples human-confirmed as hate. Finetuning a classifier on ToxiGen improves it on
**human-written** data — precedent for judge-generated synthetic hard negatives.
- **OpenAI omni-moderation-latest** ([docs](https://developers.openai.com/api/docs/guides/moderation)):
multimodal (text+image, 20MB cap), free endpoint, **13 flat categories**
(hate, hate/threatening, harassment, harassment/threatening, illicit, illicit/violent,
self-harm x3, sexual, sexual/minors, violence, violence/graphic) with per-category
scores + `category_applied_input_types`. Lesson: production taxonomies are **flat
multi-label with hierarchical-ish pairs** (base + threatening/graphic variant),
not deep trees — cheap to threshold per category.

## 2. LLM-as-judge moderation

- **Llama Guard 1** ([arXiv:2312.06674](https://arxiv.org/abs/2312.06674)): Llama2-7B
**instruction-tuned** (not prompted) on a small high-quality safety dataset; does
prompt+response classification, outputs binary decisions, matches/exceeds prior tools
on OpenAI ModEval + ToxicChat. Llama Guard 3-8B reports F1 0.939 / AUPRC 0.985 / FPR 0.040
([HF card](https://huggingface.co/meta-llama/Llama-Guard-3-8B)); Guard 4 is 12B
([Meta docs](https://dev.meta.ai/llama/docs/model-cards-and-prompt-formats/llama-guard-4));
1B-INT4 variant exists for edge ([arXiv:2411.17713](https://arxiv.org/abs/2411.17713)).
Pattern: **finetune with taxonomy in the prompt template**, so categories stay
customizable zero-shot at inference.
- **ShieldGemma** ([arXiv:2407.21772](https://arxiv.org/abs/2407.21772)): Gemma2-based,
claims +10.8% AU-PRC over Llama Guard on public benchmarks; key method is an
**LLM-based synthetic data curation pipeline** — model trained mainly on synthetic data
still generalizes. ShieldGemma 2 is 4B multimodal ([arXiv:2504.01081](https://arxiv.org/abs/2504.01081),
[HF](https://huggingface.co/google/shieldgemma-2-4b-it)). Takeaway: synthetic-first
training data is now mainstream, not a hack.
- **WildGuard** ([arXiv:2406.18495](https://arxiv.org/abs/2406.18495),
[github.com/allenai/wildguard](https://github.com/allenai/wildguard)): 7B Mistral-based,
**multi-task in one model** — prompt harm + response harm + refusal detection —
trained on 92k WildGuardMix (vanilla + adversarial jailbreaks paired with
refusal/compliance responses). Up to +26.4% on refusal detection vs open baselines,
matches/beats GPT-4 (up to +3.9% prompt-harm ID); jailbreak success 79.8%->2.4%.
Pattern most relevant to us: **one judge, three heads, adversarial pairs in training**.
- **Cost/latency handling**: nobody runs 7B judgesinline per request. Standard split is
exactly our L0-L3: cheap classifier gate synchronously, LLM judge asynchronously /
on-flag-only (OpenAI bakes moderation into the Responses API as a sidecar signal,
not a blocker — see omni docs above). Guided/constrained JSON output (e.g. Outlines-style
logit masking) is the norm for machine-readable verdicts; finetuned judges emit the
label token first so a single-token read = classification latency.

## 3. Small-from-large: distillation + quantized CPU deployment

- **DistilBERT** ([arXiv:1910.01108](https://arxiv.org/abs/1910.01108)): 6-layer student,
triple loss (distillation KL + MLM + cosine embedding), no token-type embeddings;
reports ~97% of BERT-base performance at ~40% fewer params, ~60% faster.
- **TinyBERT** ([arXiv:1909.10351](https://arxiv.org/abs/1909.10351) /
[ACL 2020.findings-emnlp.372](https://aclanthology.org/2020.findings-emnlp.372/)):
**two-stage distillation** (pretraining-stage attention/hidden-state transfer, then
task-stage) + layer mapping. TinyBERT-4 keeps 96.8% of BERT-Base on GLUE at
7.5x smaller / 9.4x faster. This two-stage recipe is the one to copy for
judge->encoder: distill **logits + hidden states**, not just hard labels.
- **ONNX dynamic int8** ([optimum-onnx docs](https://huggingface.co/docs/optimum-onnx/en/onnxruntime/usage_guides/quantization),
[ORT docs](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)):
`quantize_dynamic` (int8 weights, fp32 compute, no calibration set) via
`ORTQuantizer` + `AutoQuantizationConfig.avx512_vnni`. Teams hit CPU gates by
combining: fewer layers (xsmall/mini) + seq-len cap (96, not 512) + dynamic int8 +
single-batch graph optimization. Static int8 is ~3x faster but needs calibration
data ([Phil Schmid guide](https://www.philschmid.de/static-quantization-optimum)) —
upgrade path only if dynamic misses the gate.

## 4. Vietnamese specifics

- **ViHateT5** ([ACL 2024.findings-acl.355](https://aclanthology.org/2024.findings-acl.355),
[arXiv:2405.14141](https://arxiv.org/abs/2405.14141)): T5-base pretrained on VOZ-HSD
(large-scale Vietnamese forum crawl), then **one unified text-to-text model** for all
HSD tasks; SOTA across standard VI HSD benchmarks. Two lessons: (a) **in-domain
social pretraining >> generic Wikipedia pretraining** (mirrors HateBERT finding);
(b) multitask-unified beats fragmented per-task finetunes; label distribution in
pretraining data materially affects results.
- **ViSoBERT** ([arXiv:2310.11166](https://arxiv.org/abs/2310.11166),
[EMNLP 2023](https://aclanthology.org/2023.emnlp-main.315/)): first monolingual VI
**social-media** encoder (XLM-R arch, custom social tokenizer — raw emoji/teen-code
kept, not normalized away); beats prior SOTA on 5 VI social tasks (emotion, HSD,
sentiment, spam, span detection) with fewer params. Tokenizer lesson: keep
diacritic-less/teen-code variants as first-class tokens.
- **ViHSD** ([arXiv:2103.11528](https://arxiv.org/abs/2103.11528),
[github.com/sonlam1102/vihsd](https://github.com/sonlam1102/vihsd)): 33.4k comments,
3 labels (CLEAN / OFFENSIVE / HATE). Baselines: TextCNN/GRU + mBERT/XLM-R/DistilBERT.
- **ViCTSD** ([arXiv:2103.10069](https://arxiv.org/abs/2103.10069),
[github.com/tarudesu/ViCTSD](https://github.com/tarudesu/ViCTSD)): 10k comments over
10 domains, **dual labels** (constructive? toxic?) — seven toxicity-adjacent signal
plus span-adjacent constructiveness. PhoBERT Transfer baselines.

## Comparison

| Approach | Reported accuracy | Latency class | Cost | Our verdict |
|---|---|---|---|---|
| Multi-label encoder (Detoxify/xlm-r) | AUC 92-99 per head | ~50ms CPU int8 seq96 (our meas.) | ~0 | Too slow for L0 gate at base size; keep as L2 |
| xsmall encoder (videberta-xsmall) | MF1 0.774 (our meas.) | 14.7ms p99 int8 (our meas.) | ~0 | **L0/L1 gate**; close gap via distillation |
| Char-level Transformer (Perspective) | prod SOTA multilingual | server-side | API $ | Copy idea: char/subword-robust features for obfuscation |
| Domain-pretrained encoder (HateBERT/ViHateT5/ViSoBERT) | beats generic base | same as base | 1x pretrain | **Do this**: continue-pretrain xsmall on VOZ-style data |
| Finetuned 7-8B judge (LlamaGuard/WildGuard) | F1 0.94 / ~GPT-4 parity | seconds, GPU | $$ | Async L3 only, on-flag routing |
| Synthetic-data judge (ShieldGemma) | +10.8% AU-PRC | same | 1x curation | Endorses our judge-labels-encoder loop |
| Distilled 4-6L student (Distil/TinyBERT) | 96-97% of teacher | 7-9x faster | 1x distill | Recipe for judge->encoder transfer |

## 5 takeaways for OUR v1

1. **Gate = videberta-xsmall int8 (14.7ms), not xlm-r-base (51ms).** Close the
0.824->0.774 MF1 gap with TinyBERT-style two-stage distillation (hidden-state +
logit transfer) from the Qwen3-4B judge, not by upsizing the gate.
2. **Multi-label heads + identity auxiliary labels** (Detoxify 2019 design): add
target-group auxiliary heads and slice evals (gender/religion/region mentions) from
day one; bias slices are where production judges fail first.
3. **Continue-pretrain the encoder on in-domain VI social text** (HateBERT/ViHateT5
recipe) before any distillation — generic pretraining is the documented MF1 leak.
4. **Judge = one QLoRA model, three tasks** (WildGuard recipe): prompt-harm,
response-harm, refusal/compliance — with taxonomy in the prompt template
(Llama Guard pattern) so categories stay editable without retraining; emit the
verdict token first for single-token classification reads.
5. **Synthetic hard negatives from the judge** (ToxiGen/ShieldGemma recipe):
implicit-toxicity + obfuscated (no-diacritic, teen-code) pairs fed back into encoder
training; keep static int8 + seq-96 cap as the reserved upgrade if dynamic int8
regresses under load.
