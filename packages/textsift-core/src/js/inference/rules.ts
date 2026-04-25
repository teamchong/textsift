/**
 * Custom detection rules — regex / function matchers that run
 * alongside the model. Their output is merged into the same
 * `DetectedSpan[]` the model produces.
 *
 * Why this lives in textsift-core: rules are pure JS, no model
 * inference, no GPU/WASM. They run on the same input string the
 * model sees and emit spans in the same `start/end/label` shape, so
 * the rest of the pipeline (BIOES merge, redaction applicator,
 * streaming flush logic) works on them unchanged.
 *
 * Merge semantics with model spans:
 *   - Rule spans and model spans coexist by default.
 *   - When a rule span overlaps a model span, the rule wins (the
 *     caller wrote the rule explicitly; presumed intent is to
 *     override or refine model output).
 *   - Rule spans never overlap each other within one rule (regex
 *     matchAll already yields non-overlapping). Across rules, ties
 *     break by rule order in the array.
 */

import type { DetectedSpan, Rule, SpanLabel } from "../types.js";
import { PrivacyFilterError } from "../types.js";

/**
 * Run all rules against `input` and return the resulting spans.
 * Doesn't dedup against model spans — caller does that via
 * `mergeRuleSpans()`.
 */
export function runRules(input: string, rules: readonly Rule[]): DetectedSpan[] {
  const out: DetectedSpan[] = [];
  for (const rule of rules) {
    const matches = collectMatches(input, rule);
    for (const m of matches) {
      if (m.start >= m.end) continue;
      if (m.start < 0 || m.end > input.length) continue;
      out.push({
        label: rule.label as SpanLabel | string,
        source: "rule",
        severity: rule.severity ?? "warn",
        start: m.start,
        end: m.end,
        text: input.slice(m.start, m.end),
        marker: rule.marker ?? `[${rule.label}]`,
        confidence: 1.0,
      });
    }
  }
  out.sort((a, b) => a.start - b.start);
  return out;
}

function collectMatches(
  input: string,
  rule: Rule,
): Array<{ start: number; end: number }> {
  if ("pattern" in rule) {
    if (!rule.pattern.global) {
      throw new PrivacyFilterError(
        `Rule "${rule.label}": pattern must be a global regex (use the /g flag); ` +
          `got ${String(rule.pattern)}`,
        "INTERNAL",
      );
    }
    const out: Array<{ start: number; end: number }> = [];
    for (const m of input.matchAll(rule.pattern)) {
      const start = m.index ?? 0;
      const len = m[0].length;
      if (len === 0) continue;
      out.push({ start, end: start + len });
    }
    return out;
  }
  return rule.match(input);
}

/**
 * Merge model spans + rule spans into a single ordered list.
 * Rule spans win on overlap; identical-range pairs prefer the rule
 * (since the caller authored it deliberately).
 */
export function mergeRuleSpans(
  modelSpans: readonly DetectedSpan[],
  ruleSpans: readonly DetectedSpan[],
): DetectedSpan[] {
  if (ruleSpans.length === 0) return [...modelSpans];

  const out: DetectedSpan[] = [];
  // Walk both lists in start order. When a rule span overlaps a model
  // span, drop the model span (and any later model spans whose start
  // is still within the rule span's end).
  const model = [...modelSpans].sort((a, b) => a.start - b.start);
  const rules = [...ruleSpans].sort((a, b) => a.start - b.start);
  let mi = 0;
  let ri = 0;
  while (mi < model.length || ri < rules.length) {
    const m = model[mi];
    const r = rules[ri];
    if (r === undefined) {
      out.push(model[mi]!);
      mi++;
      continue;
    }
    if (m === undefined) {
      out.push(rules[ri]!);
      ri++;
      continue;
    }
    if (overlaps(m, r)) {
      // Rule wins; emit it, advance ri, and skip every model span
      // it covers.
      out.push(r);
      ri++;
      while (mi < model.length && model[mi]!.start < r.end) mi++;
      continue;
    }
    if (m.start <= r.start) {
      out.push(m);
      mi++;
    } else {
      out.push(r);
      ri++;
    }
  }
  return out;
}

function overlaps(a: DetectedSpan, b: DetectedSpan): boolean {
  return a.start < b.end && b.start < a.end;
}
