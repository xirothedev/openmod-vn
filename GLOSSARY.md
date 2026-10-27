# Glossary

Ubiquitous language for openmod-vn. Implementation details live in code and ADRs, not here.

| Term | Meaning |
| ---- | ------- |
| Explicit profanity | Single term offensive on its own, no context needed (e.g. slur). Routes to fast block or mask. |
| Context-dependent word | Benign alone (animal, object), offensive when aimed at a person (e.g. `chó` as insult). Routes to Decision Engine. |
| Compositional toxicity | Benign single words (`w1, w2 in Clean`) that form an offensive, sexual, or dehumanizing meaning when combined (`chăn rau`). |
| Candidate span | Suspicious substring from L1 or L2 that needs a decision. Carries normalized span and original span. |
| Suspicion score | L2 output 0.0-1.0. Positive context lowers it, toxic context raises it. `>= threshold` triggers L3. |
| Contrastive pair | Two benign plus two toxic examples using the same tokens. Required for every compositional rule. |
| Decision | Final output: `ALLOW \| MASK \| WARN \| BLOCK \| ESCALATE` plus severity 0-4, category, spans, rationale. |
| Encoder label | L3 encoder output: `BLOCK \| BORDERLINE \| ALLOW` plus spans. Heuristic maps it to Decision. |
| Scunthorpe defense | Whitelist check that stops substring matches inside clean words. |
| Policy profile | Preset thresholds per surface (`kid_safe`, `standard_social`, `relaxed_gaming`). |
