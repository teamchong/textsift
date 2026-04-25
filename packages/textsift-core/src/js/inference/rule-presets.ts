/**
 * Curated rule presets — high-precision detectors for things the
 * model wasn't trained on but a proxy/CI/IDE plugin probably wants
 * to block.
 *
 * Selection criteria:
 *   - Distinct prefix or fixed structure (low false-positive rate)
 *   - High stakes if leaked (credential, API key, private key)
 *   - Disjoint from the model's PII categories — these complement,
 *     they don't duplicate
 *
 * Severity defaults to `"block"` because every entry here is something
 * a sane policy would refuse to forward (vs `"warn"` for fuzzier
 * categories like JIRA tickets or internal hostnames). Callers can
 * override per-rule by spreading and reshaping.
 */

import type { Rule } from "../types.js";

/**
 * Named credential / secret patterns. Includes:
 *   - JWT (header always starts with `eyJ` = Base64 of `{"`)
 *   - GitHub Personal Access Tokens (all variants: ghp_, gho_, ghu_, ghs_, ghr_)
 *   - GitHub fine-grained PAT (`github_pat_...`)
 *   - AWS Access Key ID (root + temp)
 *   - Slack bot/user/webhook tokens
 *   - OpenAI API keys (legacy + project)
 *   - Anthropic API keys
 *   - Google API keys
 *   - Stripe live/test secret + publishable + restricted + webhook
 *   - NPM tokens
 *   - PEM-formatted private key headers (RSA, EC, OPENSSH, DSA, PGP)
 */
export const secretRules: readonly Rule[] = Object.freeze([
  {
    label: "JWT",
    severity: "block",
    marker: "[JWT]",
    pattern: /\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b/g,
  },
  {
    label: "GITHUB_PAT_CLASSIC",
    severity: "block",
    marker: "[GITHUB_TOKEN]",
    pattern: /\bgh[opusr]_[A-Za-z0-9]{36}\b/g,
  },
  {
    label: "GITHUB_PAT_FINE_GRAINED",
    severity: "block",
    marker: "[GITHUB_TOKEN]",
    pattern: /\bgithub_pat_[A-Za-z0-9_]{82}\b/g,
  },
  {
    label: "AWS_ACCESS_KEY_ID",
    severity: "block",
    marker: "[AWS_KEY]",
    pattern: /\b(?:AKIA|ASIA)[0-9A-Z]{16}\b/g,
  },
  {
    label: "SLACK_TOKEN",
    severity: "block",
    marker: "[SLACK_TOKEN]",
    pattern: /\bxox[bparso]-\d{10,13}-\d{10,13}(?:-\d{10,13})?-[a-f0-9]{32,}\b/g,
  },
  {
    label: "SLACK_WEBHOOK",
    severity: "block",
    marker: "[SLACK_WEBHOOK]",
    pattern: /\bhttps:\/\/hooks\.slack\.com\/services\/T[A-Z0-9]+\/B[A-Z0-9]+\/[A-Za-z0-9]+\b/g,
  },
  {
    label: "OPENAI_API_KEY",
    severity: "block",
    marker: "[OPENAI_KEY]",
    pattern: /\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b/g,
  },
  {
    label: "ANTHROPIC_API_KEY",
    severity: "block",
    marker: "[ANTHROPIC_KEY]",
    pattern: /\bsk-ant-(?:api\d{2}-)?[A-Za-z0-9_-]{90,}\b/g,
  },
  {
    label: "GOOGLE_API_KEY",
    severity: "block",
    marker: "[GOOGLE_KEY]",
    pattern: /\bAIza[A-Za-z0-9_-]{35}\b/g,
  },
  {
    label: "STRIPE_KEY",
    severity: "block",
    marker: "[STRIPE_KEY]",
    pattern: /\b(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{24,}\b/g,
  },
  {
    label: "STRIPE_WEBHOOK_SECRET",
    severity: "block",
    marker: "[STRIPE_WEBHOOK]",
    pattern: /\bwhsec_[A-Za-z0-9]{32,}\b/g,
  },
  {
    label: "NPM_TOKEN",
    severity: "block",
    marker: "[NPM_TOKEN]",
    pattern: /\bnpm_[A-Za-z0-9]{36}\b/g,
  },
  {
    label: "PRIVATE_KEY",
    severity: "block",
    marker: "[PRIVATE_KEY]",
    pattern: /-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----/g,
  },
]);

/** Map from preset name → rule set. New presets get added here. */
export const RULE_PRESETS: Readonly<Record<string, readonly Rule[]>> = Object.freeze({
  secrets: secretRules,
});

/** Names callers can pass to the `presets` option. */
export type RulePresetName = keyof typeof RULE_PRESETS;

/**
 * Resolve a `presets` option to its rule set. Unknown preset names
 * throw — no silent ignores, since a typo would mean the caller
 * thinks they're protected when they aren't.
 */
export function resolvePresets(presets: readonly string[]): Rule[] {
  const out: Rule[] = [];
  for (const name of presets) {
    const set = RULE_PRESETS[name as RulePresetName];
    if (!set) {
      throw new Error(
        `Unknown rule preset "${name}". Available: ${Object.keys(RULE_PRESETS).join(", ")}`,
      );
    }
    out.push(...set);
  }
  return out;
}
