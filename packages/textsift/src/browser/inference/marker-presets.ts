/**
 * Built-in marker strategies, exported as `markerPresets` from the
 * package root. The opposite kind of "preset" from `RULE_PRESETS` —
 * those add detection rules, these change how detected spans are
 * replaced in the output.
 *
 * `markerPresets.faker` is a factory: each call returns a fresh
 * stateful `MarkerStrategy` that produces realistic-but-fake values.
 * Within a single strategy instance the same input text always maps
 * to the same fake output (so `John Smith` mentioned three times
 * becomes the same fake name three times — preserving relationships
 * across mentions for downstream tooling that cares).
 *
 * Use case: generating prod-shaped test fixtures, sales demos,
 * customer support transcript sharing with vendors — anywhere you
 * want the redacted output to still look like real data instead of
 * `[private_email]`-style markers that break downstream validators.
 */

import type { DetectedSpan, MarkerStrategy, SpanLabel } from "../types.js";

/**
 * Pool of synthesised values per category. Picked round-robin (mod
 * length) per category instance. Pools are small but provide enough
 * variety that realistic test docs don't all share two names. The
 * intent is plausibility for downstream code that pattern-matches on
 * shape, not cryptographic novelty — these are fakes, not anonymised
 * real data.
 *
 * Notable omission: `secret`. Faking a secret with another
 * credible-looking secret is a footgun (downstream code might treat
 * it as real, or attackers might pattern-mine the leaked-but-fake
 * format). `secret` spans always emit `[secret]` regardless of
 * preset.
 */
const PERSON_POOL = [
  "Alice Anderson",
  "Maria Rodriguez",
  "James Chen",
  "Olivia Patel",
  "Liam Nakamura",
  "Sophia Williams",
  "Noah Becker",
  "Emma Dubois",
  "Lucas Kowalski",
  "Mia Johansson",
  "Ethan Tanaka",
  "Ava Hernandez",
] as const;

function emailFromName(person: string): string {
  return `${person.toLowerCase().replace(/[^a-z]+/g, ".")}@example.com`;
}

function pad(n: number, width: number): string {
  return n.toString().padStart(width, "0");
}

interface FakeContext {
  counts: Map<SpanLabel | string, number>;
}

function nextIndex(ctx: FakeContext, label: SpanLabel | string): number {
  const cur = ctx.counts.get(label) ?? 0;
  ctx.counts.set(label, cur + 1);
  return cur;
}

function generateFake(span: DetectedSpan, ctx: FakeContext): string {
  const label = span.label;
  const i = nextIndex(ctx, label);
  switch (label) {
    case "private_person": {
      const idx = i % PERSON_POOL.length;
      return PERSON_POOL[idx]!;
    }
    case "private_email": {
      const idx = i % PERSON_POOL.length;
      return emailFromName(PERSON_POOL[idx]!);
    }
    case "private_phone":
      // Reserved-for-fiction US prefix (NANPA "555-01XX" range).
      return `+1-555-01${pad(i % 100, 2)}`;
    case "private_address":
      // Generic format that parses as an address but maps nowhere.
      return `${100 + i} Main St, Springfield, IL ${pad(60000 + i, 5)}`;
    case "private_url":
      // Reserved for documentation per RFC 2606 / 6761.
      return `https://example.com/path/${i}`;
    case "private_date":
      // Walk through 2026 in 7-day steps; cosmetic, not parseable
      // back to the original date.
      return new Date(Date.UTC(2026, 0, 1) + i * 7 * 86400_000)
        .toISOString().slice(0, 10);
    case "account_number":
      // Test-only Visa BIN (4111-1111-1111-XXXX is the canonical one
      // for "this is a test card, do not charge").
      return `4111-1111-1111-${pad(i % 10000, 4)}`;
    case "secret":
      // Don't synthesise. See note above.
      return "[secret]";
    default:
      // Unknown / custom-rule label: fall back to the default `[label]`.
      return `[${label}]`;
  }
}

/**
 * Built-in marker strategies. Pass `markers: markerPresets.faker()`
 * to `PrivacyFilter.create()` to swap `[private_email]`-style markers
 * for realistic-looking fake values, consistent within the strategy
 * instance's lifetime.
 *
 * ```ts
 * import { PrivacyFilter, markerPresets } from "textsift";
 *
 * const filter = await PrivacyFilter.create({
 *   markers: markerPresets.faker(),
 * });
 * await filter.redact("Hi John, your email is john@example.com");
 * // → "Hi Alice Anderson, your email is alice.anderson@example.com"
 *
 * // Mention "John" again later — same fake comes back, preserving
 * // the relationship across the document:
 * await filter.redact("John signed up on 2025-12-01");
 * // → "Alice Anderson signed up on 2026-01-01"
 * ```
 */
export const markerPresets = Object.freeze({
  /**
   * Faker mode: emit realistic-looking fake values instead of
   * `[label]` markers. Useful for generating test fixtures, sales
   * demos, or anywhere downstream code expects PII-shaped data even
   * after redaction.
   *
   * Returns a fresh `MarkerStrategy` per call so each filter gets
   * its own consistency state. Call once at filter creation.
   */
  faker(): MarkerStrategy {
    const ctx: FakeContext = { counts: new Map() };
    const memo = new Map<string, string>();
    return (span: DetectedSpan, _index: number): string => {
      const key = `${span.label}|${span.text}`;
      const cached = memo.get(key);
      if (cached !== undefined) return cached;
      const fake = generateFake(span, ctx);
      memo.set(key, fake);
      return fake;
    };
  },
});

export type MarkerPresetName = keyof typeof markerPresets;
