import { decideLite, env, loadBundle, remoteModerate, type Policy } from "./moderation.js";
import { WordFilter } from "./filter.js";

const CASES: { text: string; want: string }[] = [
  { text: "Cho tôi một bát bún bò Huế.", want: "ALLOW (whitelist bún)" },
  { text: "Con chó nhà tôi rất ngoan.", want: "ALLOW (context-dependent, benign)" },
  { text: "Thằng này địt cả lò nhà mày.", want: "delete+reply (explicit)" },
  { text: "Đồ con chó, cút khỏi đây.", want: "delete+reply (aimed at person)" },
  { text: "Nông dân trồng rau sạch ngoài vườn.", want: "ALLOW (compositional benign)" },
  { text: "Đi chăn rau khu Cầu Giấy, cho xin số điện thoại em nhé?", want: "delete (compositional toxic)" },
];

const POLICIES: Policy[] = ["standard_social", "kid_safe", "relaxed_gaming"];

async function main(): Promise<void> {
  const mode = env("MODERATION_MODE") === "REMOTE_API" ? "REMOTE_API" : "OFFLINE_LITE";
  console.log(`mode=${mode}`);
  const bundle = mode === "OFFLINE_LITE" ? await loadBundle() : null;
  if (bundle) {
    console.log(`lexicon=${bundle.lexicon.length} rules=${bundle.rules.length} whitelist=${bundle.whitelist.size}`);
  }
  const filter = new WordFilter();
  await filter.init();

  const rawBase =
    env("GITHUB_RAW_BASE") ??
    "https://raw.githubusercontent.com/xirothedev/openmod-vn/main/data";
  const [l3preview, bench] = await Promise.all([
    fetch(`${rawBase}/l3_vi.jsonl`).then(async (r) => (await r.text()).split("\n").filter(Boolean).slice(0, 3)),
    fetch(`${rawBase}/benchmarks/vihos_spans.jsonl`).then(async (r) => (await r.text()).split("\n").filter(Boolean).slice(0, 2)),
  ]);
  console.log(`l3_vi preview rows=${l3preview.length} (classifier training shape)`);
  console.log(`benchmarks sample rows=${bench.length}`);

  for (const policy of POLICIES) {
    console.log(`\n## policy=${policy}`);
    for (const c of CASES) {
      if (mode === "REMOTE_API") {
        const d = await remoteModerate(c.text, policy);
        console.log(`- [${d.action}] ${JSON.stringify(c.text)} :: ${c.want}`);
      } else {
        const d = decideLite(c.text, bundle!, policy);
        const r = await filter.handle(c.text, policy);
        console.log(`- [${d.action} delete=${r.delete}] ${JSON.stringify(c.text)} :: ${c.want}`);
      }
    }
  }
}

main().catch((e) => {
  console.error(e);
  (globalThis as { process?: { exit: (c: number) => void } }).process?.exit(1);
});
