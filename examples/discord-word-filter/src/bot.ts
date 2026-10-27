import { env } from "./moderation.js";
import { WordFilter } from "./filter.js";

/**
 * Minimal discord.js wiring. discord.js is loaded lazily so `demo.ts` runs
 * without installing it; install + set DISCORD_TOKEN to run the real bot.
 */
export async function startBot(): Promise<void> {
  const token = env("DISCORD_TOKEN");
  if (!token) throw new Error("DISCORD_TOKEN is not set");
  const discord = (await import("discord.js")) as unknown as {
    Client: new (opts: unknown) => {
      on: (ev: string, fn: (msg: { author: { bot: boolean }; content: string; delete: () => Promise<unknown>; channel: { send: (t: string) => Promise<unknown> } }) => Promise<void>) => void;
      login: (t: string) => Promise<unknown>;
    };
    GatewayIntentBits: Record<string, number>;
  };
  const filter = new WordFilter();
  await filter.init();
  const client = new discord.Client({
    intents: [discord.GatewayIntentBits.Guilds, discord.GatewayIntentBits.GuildMessages, discord.GatewayIntentBits.MessageContent],
  });
  client.on("messageCreate", async (msg) => {
    if (msg.author.bot) return;
    const r = await filter.handle(msg.content);
    if (r.delete) {
      await msg.delete().catch(() => undefined);
      if (r.reply) await msg.channel.send(r.reply).catch(() => undefined);
    }
  });
  await client.login(token);
}

if (env("DISCORD_TOKEN")) {
  startBot().catch((e) => {
    console.error(e);
    (globalThis as { process?: { exit: (c: number) => void } }).process?.exit(1);
  });
}
