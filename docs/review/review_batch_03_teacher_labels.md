# Review batch 03 — teacher labels (ViHateT5, UNREVIEWED)

Method: `tarudesu/ViHateT5-base-HSD` offline on `data/_staging/vihos_{train,dev}.csv`.
Map: HATE→BLOCK, OFFENSIVE→BORDERLINE, CLEAN→ALLOW. Spans: T5 `[hate]` marks,
fallback staging indices. Teachers research-only, weights never bundled.
Files: local mirror `~/.cache/openmod-review/l3_teacher_{train,dev}.jsonl` (outside repo on
purpose: validator gates `data/**/*.jsonl`, UNREVIEWED rows must not sit under `data/`)
+ bvpt3 `~/openmod-train/review/`.

Counts (train 8844 / dev 1106): CLEAN 5181/670, OFFENSIVE 1263/139, HATE 2400/297.
Zero UNKNOWN. 692 CLEAN rows carry spans. 33 HATE/OFFENSIVE rows have no spans.

## Slice decisions (reply ACCEPT / DROP / HOLD per slice)

- [ ] S1 CLEAN no-span (4489 rows) → ACCEPT as ALLOW. Proposal: ACCEPT.
- [ ] S2 HATE/OFFENSIVE with spans (3630 rows) → ACCEPT pending spot-checks below. Proposal: ACCEPT.
- [ ] S3 CLEAN with spans (692 rows) → spans on CLEAN rows are suspect. Proposal: STRIP spans, keep ALLOW.
- [ ] S4 HATE/OFFENSIVE no-span (33 rows) → weak rows, no span signal. Proposal: DROP.
- [ ] S5 political content (e.g. `t5_00201` phản động) → policy-sensitive. Proposal: HOLD for policy call.

## Spot-checks (reply approve/reject per line)

HATE:
- [ ] `t5_08232` Câm mồm k thằng ml này → spans [câm mồm, thằng ml]
- [ ] `t5_05520` dkm thằng nào sủa câu này -_- → spans [dkm, thằng, sủa]
- [ ] `t5_07106` Chừng nào cái bọn súc sinh này mới tha cho dân VN đây → spans [bọn súc sinh]
- [ ] `t5_01898` Tổ chức đám tang . Toang rồi . → spans [Toang] (suspect: mourning context?)
- [ ] `t5_00201` Ở với + sản.. phản động.. (political, see S5)

OFFENSIVE:
- [ ] `t5_06744` Trường hợp này ... gọi là khùng á → spans [khùng] (suspect: disability-adjacent?)
- [ ] `t5_03199` Quá troi xl , đm bán hàng 10 năm , bớt đi cd oi → spans [xl, đm, cd]
- [ ] `t5_02667` Bê Lol chạy xung quanh. → spans [lol]
- [ ] `t5_07864` Rồi, rớt cái lòn luôn → spans [lòn]

CLEAN:
- [ ] `t5_06681` đây là doanh nghiệp đề nghị nhé, chứ chính quyền chưa đề nghị đâu → no spans
- [ ] `t5_02167` Ăn nên làm ra là phải ĐẦU TƯ 👏🏿👏🏿👏🏿👏🏿 → no spans
- [ ] `t5_07334` Sao t gửi đc bây → no spans

## Gate

After approval: build `data/l3_vi.jsonl` from approved slices/rows only, validator green
(source/source_url/license/retrieved_at per row), then Step 3 encoder train.
