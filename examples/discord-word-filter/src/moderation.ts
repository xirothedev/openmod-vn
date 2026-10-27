/**
 * Offline-lite moderation port of apps/core L0-L2 + heuristic L3.
 * REMOTE_API (full L0-L3 incl. ViGuard-X shadow) lives in remoteModerate().
 * Data: fetch via GitHub raw (data/*.jsonl) or local dist/*.json (release download).
 */

export type Policy = "kid_safe" | "standard_social" | "relaxed_gaming";
export type Action = "ALLOW" | "MASK" | "WARN" | "BLOCK" | "ESCALATE";

export interface Span {
  start: number;
  end: number;
  text: string;
  category: string;
  is_compositional: boolean;
  explanation: string;
}

export interface Decision {
  action: Action;
  overall_severity: number;
  detected_spans: Span[];
  masked_text: string;
  rationale: string;
}

export interface LexItem {
  id: string;
  term: string;
  variants?: string[];
  severity: number;
  requires_context_check?: boolean;
}

export interface CompRule {
  id: string;
  tokens: string[];
  max_skip_gram_distance?: number;
  category?: string;
  base_severity: number;
  toxic_context_indicators?: string[];
  positive_context_indicators?: string[];
}

const POLICIES: Record<Policy, { escalate: number; block: number; warn: number }> = {
  kid_safe: { escalate: 3, block: 2, warn: 1 },
  standard_social: { escalate: 4, block: 3, warn: 2 },
  relaxed_gaming: { escalate: 5, block: 4, warn: 3 },
};

const AIMED_CUES = ["đồ", "lũ", "tụi", "bọn", "mày", "thằng", "cút", "chửi"];

const HOMOGLYPH_FALLBACK: Record<string, string> = {
  "0": "o",
  "1": "i",
  "3": "e",
  "4": "a",
  "5": "s",
  "6": "g",
  "7": "t",
  "8": "b",
  "9": "g",
  "@": "a",
  $: "s",
};

/** Read env without needing @types/node (examples stay dependency-free). */
export function env(name: string): string | undefined {
  try {
    const g = globalThis as unknown as { process?: { env?: Record<string, string | undefined> } };
    return g.process?.env?.[name] ?? undefined;
  } catch {
    return undefined;
  }
}

export const RAW_BASE = env("GITHUB_RAW_BASE") ??
  "https://raw.githubusercontent.com/xirothedev/openmod-vn/main/data";

async function fetchJsonl(url: string, limit = Infinity): Promise<unknown[]> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url}: ${res.status}`);
  const text = await res.text();
  const out: unknown[] = [];
  for (const line of text.split("\n")) {
    const t = line.trim();
    if (!t) continue;
    out.push(JSON.parse(t));
    if (out.length >= limit) break;
  }
  return out;
}

async function fetchJson(url: string): Promise<unknown> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url}: ${res.status}`);
  return res.json() as Promise<unknown>;
}

export interface Bundle {
  lexicon: LexItem[];
  whitelist: Set<string>;
  rules: CompRule[];
  teencode: Record<string, string>;
  homoglyph: Record<string, string>;
}

/** Load all runtime slices: lexicon + whitelist + compositional + normalization. */
export async function loadBundle(rawBase = RAW_BASE): Promise<Bundle> {
  const [explicit, imported, white, rules, teen, homo] = await Promise.all([
    fetchJsonl(`${rawBase}/raw_lexicons/vi/explicit_profanity.jsonl`),
    fetchJsonl(`${rawBase}/raw_lexicons/vi/offensive_words_import.jsonl`),
    fetchJsonl(`${rawBase}/raw_lexicons/vi/safe_whitelist.jsonl`),
    fetchJsonl(`${rawBase}/compositional/vi_compositional_phrases.jsonl`),
    fetchJson(`${rawBase}/normalization/teencode_mapping.json`),
    fetchJson(`${rawBase}/normalization/leetspeak_homoglyphs.json`).catch(() => ({})),
  ]);
  const lexicon = [...(explicit as LexItem[]), ...(imported as LexItem[])];
  const whitelist = new Set(
    (white as { term: string }[]).map((w) => w.term.toLowerCase()),
  );
  return {
    lexicon,
    whitelist,
    rules: rules as CompRule[],
    teencode: teen as Record<string, string>,
    homoglyph: { ...HOMOGLYPH_FALLBACK, ...((homo as Record<string, string>) ?? {}) },
  };
}

/** L0-lite: NFC + zero-width strip + teencode word lookup + homoglyph fold. */
export function normalizeLite(
  text: string,
  teencode: Record<string, string>,
  homoglyph: Record<string, string>,
): string {
  let s = text.normalize("NFC").replace(/[\u200b\u200c\u200d\ufeff\u2060\u00ad]/g, "");
  const words = s.split(/(\s+)/);
  const mapped = words.map((w) => {
    if (/^\s+$/.test(w) || w === "") return w;
    const folded = [...w.toLowerCase()].map((c) => homoglyph[c] ?? c).join("");
    return teencode[folded] ?? folded;
  });
  s = mapped.join("").replace(/(.)\1{2,}/g, "$1$1");
  return s.replace(/\s+/g, " ").trim();
}

function isBoundary(norm: string, s: number, e: number): boolean {
  const left = s === 0 || !/[a-z0-9à-ỹ]/i.test(norm[s - 1] ?? "");
  const right = e >= norm.length || !/[a-z0-9à-ỹ]/i.test(norm[e] ?? "");
  return left && right;
}

function maskText(text: string, spans: Span[]): string {
  const sorted = [...spans].sort((a, b) => b.start - a.start);
  let out = text;
  for (const s of sorted) out = out.slice(0, s.start) + "***" + out.slice(s.end);
  return out;
}

/**
 * Heuristic decide (L1+L2 -> 5-way). Mirrors HeuristicContextJudge.decide.
 * L3 encoder (ViGuard-X) only runs server-side; REMOTE_API covers it.
 */
export function decideLite(text: string, bundle: Bundle, policy: Policy = "standard_social"): Decision {
  const norm = normalizeLite(text, bundle.teencode, bundle.homoglyph);
  const low = norm.toLowerCase();
  const spans: Span[] = [];
  let severity = 0;

  // L1 lexical (term + variants, whitelist suppression, context check)
  for (const p of bundle.lexicon) {
    if (bundle.whitelist.has(p.term.toLowerCase())) continue;
    const cands = [p.term, ...(p.variants ?? [])].map((t) => t.toLowerCase());
    for (const c of cands) {
      let i = low.indexOf(c);
      while (i !== -1) {
        const e = i + c.length;
        if (isBoundary(low, i, e)) {
          // lite port: norm coords reused on original (approx when folding changes length)
          const s0 = Math.min(text.length, i);
          const e0 = Math.min(text.length, e);
          if (p.requires_context_check) {
            const aimed = AIMED_CUES.some((k) => low.includes(k));
            if (aimed) {
              spans.push({
                start: s0,
                end: e0,
                text: text.slice(s0, e0),
                category: "INSULT_DEHUMANIZATION",
                is_compositional: false,
                explanation: `context-dependent term aimed at person: ${p.term}`,
              });
              severity = Math.max(severity, p.severity);
            }
          } else {
            spans.push({
              start: s0,
              end: e0,
              text: text.slice(s0, e0),
              category: "PROFANITY",
              is_compositional: false,
              explanation: `explicit match: ${p.term}`,
            });
            severity = Math.max(severity, p.severity);
          }
          break; // one hit per pattern keeps demo output stable
        }
        i = low.indexOf(c, i + 1);
      }
    }
  }

  // L2 compositional skip-gram
  const toks = norm.split(/\s+/).filter(Boolean);
  for (const r of bundle.rules) {
    const want = r.tokens.map((t) => t.toLowerCase());
    const gap = r.max_skip_gram_distance ?? 2;
    let found: [number, number] | null = null;
    for (let i = 0; i < toks.length && !found; i++) {
      if (toks[i].toLowerCase() !== want[0]) continue;
      let j = i;
      let k = 1;
      while (k < want.length) {
        let next = -1;
        for (let d = 1; d <= gap + 1; d++) {
          if (j + d < toks.length && toks[j + d].toLowerCase() === want[k]) {
            next = j + d;
            break;
          }
        }
        if (next === -1) break;
        j = next;
        k++;
      }
      if (k === want.length) found = [i, j];
    }
    if (!found) continue;
    let score = 0.3;
    if ((r.toxic_context_indicators ?? []).some((t) => low.includes(t.toLowerCase()))) score += 0.35;
    if ((r.positive_context_indicators ?? []).some((t) => low.includes(t.toLowerCase()))) score -= 0.25;
    if (score < 0.5) continue;
    const approx = toks.slice(found[0], found[1] + 1).join(" ");
    // lite port: compositional span is norm-space (approx); server returns exact orig_span
    spans.push({
      start: 0,
      end: Math.min(text.length, approx.length),
      text: approx,
      category: r.category ?? "HARASSMENT",
      is_compositional: true,
      explanation: `compositional ${r.id} suspicion=${score.toFixed(2)}`,
    });
    severity = Math.max(severity, r.base_severity);
  }

  if (spans.length === 0) {
    return {
      action: "ALLOW",
      overall_severity: 0,
      detected_spans: [],
      masked_text: text,
      rationale: "clean",
    };
  }
  const th = POLICIES[policy] ?? POLICIES.standard_social;
  const action: Action =
    severity >= th.escalate
      ? "ESCALATE"
      : severity >= th.block
        ? "BLOCK"
        : severity >= th.warn
          ? "WARN"
          : "MASK";
  return {
    action,
    overall_severity: severity,
    detected_spans: spans,
    masked_text: maskText(text, spans),
    rationale: `${spans.length} span(s), max severity ${severity}`,
  };
}

/** Full pipeline via running Python API (covers L3 ViGuard-X shadow/veto). */
export async function remoteModerate(
  text: string,
  policy: Policy = "standard_social",
  apiUrl = env("MODERATION_API_URL") ?? "http://localhost:8000",
): Promise<Decision> {
  const res = await fetch(`${apiUrl}/moderate`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ text, policy }),
  });
  if (!res.ok) throw new Error(`POST /moderate: ${res.status}`);
  return (await res.json()) as Decision;
}
