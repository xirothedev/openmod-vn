import { decideLite, env, loadBundle, remoteModerate, type Bundle, type Policy } from "./moderation.js";

/**
 * Discord word filter. `handle()` returns null when the message is clean,
 * otherwise the warning text to reply with (and the caller deletes the message).
 */
export class WordFilter {
  private bundle: Bundle | null = null;

  async init(): Promise<void> {
    if ((env("MODERATION_MODE") ?? "OFFLINE_LITE") === "OFFLINE_LITE") {
      this.bundle = await loadBundle();
    }
  }

  async handle(
    text: string,
    policy: Policy = "standard_social",
  ): Promise<{ delete: boolean; reply: string | null; action: string }> {
    const d =
      this.bundle == null
        ? await remoteModerate(text, policy)
        : decideLite(text, this.bundle, policy);
    if (d.action === "ALLOW") return { delete: false, reply: null, action: d.action };
    if (d.action === "MASK" || d.action === "WARN") {
      return { delete: true, reply: `Tin nhắn đã bị gỡ (mức ${d.action}). Vui lòng dùng từ ngữ phù hợp.`, action: d.action };
    }
    return { delete: true, reply: `Tin nhắn vi phạm (mức ${d.action}, sev=${d.overall_severity}) và đã được ghi nhận.`, action: d.action };
  }
}
