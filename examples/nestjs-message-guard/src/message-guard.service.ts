import { decideLite, env, loadBundle, remoteModerate, type Bundle, type Policy } from "./moderation.js";

/**
 * Drop-in NestJS service. Copy into a NestJS app and provide via a module:
 *   providers: [MessageGuardService], exports: [MessageGuardService]
 * Compatible with `CanActivate`: guard calls `check()` and blocks when action
 * is BLOCK or ESCALATE (kid_safe blocks from severity >= 2).
 */
export class MessageGuardService {
  private bundle: Bundle | null = null;
  private mode: "REMOTE_API" | "OFFLINE_LITE";

  constructor(mode: "REMOTE_API" | "OFFLINE_LITE" = "OFFLINE_LITE") {
    const m = env("MODERATION_MODE");
    this.mode = m === "REMOTE_API" ? "REMOTE_API" : mode;
  }

  async init(): Promise<void> {
    if (this.mode === "OFFLINE_LITE") this.bundle = await loadBundle();
  }

  /** Returns true when the message may pass. */
  async canPass(text: string, policy: Policy = "standard_social"): Promise<boolean> {
    const d =
      this.mode === "REMOTE_API" || !this.bundle
        ? await remoteModerate(text, policy)
        : decideLite(text, this.bundle, policy);
    return d.action !== "BLOCK" && d.action !== "ESCALATE";
  }

  async check(text: string, policy: Policy = "standard_social") {
    if (this.mode === "REMOTE_API" || !this.bundle) return remoteModerate(text, policy);
    return decideLite(text, this.bundle, policy);
  }
}
