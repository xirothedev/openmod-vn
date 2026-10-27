# Sources

Every record in `data/**/*.jsonl` must carry `source`, `source_url`, `license`, `retrieved_at`. This file is the human ledger.

| Data | Source | License | Use |
| ---- | ------ | ------- | --- |
| `data/compositional/*`, `data/benchmarks/*` seed v0 | `synthetic-seed-v0` (written for this repo) | CC-BY-SA-4.0 | Seed contrastive rules, human-review before release |
| `data/raw_lexicons/vi/*.jsonl` seed v0 | `synthetic-seed-v0` (written for this repo) | CC-BY-SA-4.0 | Seed lexicon + whitelist, human-review before release |
| `data/raw_lexicons/vi/offensive_words_import.jsonl` (430 terms) | https://github.com/blue-eyes-vn/vietnamese-offensive-words | MIT | Imported, all `requires_context_check=true` + `needs-review`, human-verify before BLOCK |
| `data/benchmarks/vihos_spans.jsonl` (span rows) | https://github.com/phusroyal/vihos | MIT | Imported span rows as OFFENSIVE seeds, label noise possible, review before release |
| `data/l3_vi.jsonl` (9771 rows, review batch 03) | ViHOS texts (MIT, above) teacher-labeled offline by ViHateT5 (research-only, weights never bundled), human-approved slices only | MIT (source texts) | L3 encoder train; 147 political rows held out, 32 weak rows dropped |
| `data/benchmarks/hatecheck_vi_templates.jsonl` (18) | HateCheck adapted to Vietnamese, https://github.com/paul-rottger/hatecheck-data | CC-BY-4.0 | Fresh VI labels, cite Röttger et al. 2021 |
| Raw downloads (not bundled) | `data/_staging/` (gitignored) | per-source | Methodology reference only |
| `data/raw_lexicons/vi/safe_whitelist.jsonl` | `synthetic-seed-v0` | CC-BY-SA-4.0 | Scunthorpe defense |
| Span methodology | Hoang et al. 2023 ViHOS https://arxiv.org/abs/2301.10186 | MIT (code+data) | Span schema reference, not bundled |
| Functional test methodology | Röttger et al. 2021 HateCheck https://arxiv.org/abs/2012.15606 | CC-BY-4.0 | Template structure adapted to VI, re-labeled |
| Eval-only (not bundled) | Luu et al. 2021 ViHSD https://arxiv.org/abs/2103.11528; Nguyen et al. 2021 ViCTSD https://arxiv.org/abs/2103.10069; Thanh Nguyen 2024 ViHateT5 https://arxiv.org/abs/2405.14141 | research-only / no license | Eval with citation, no redistribution |
| Preprocessing-only | Nguyen et al. 2024 ViLexNorm https://arxiv.org/abs/2401.16403 | CC-BY-NC-SA-4.0 | Do not bundle text into BY-SA release |
