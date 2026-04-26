/**
 * Apply redaction markers to an input string given detected spans.
 *
 * Guarantees:
 *   - Character-level correctness: the returned `redactedText` has each
 *     span's character range replaced with the marker resolved by the
 *     marker strategy.
 *   - Stable order: markers are applied from LAST span to FIRST, so
 *     earlier offsets remain valid while we iterate.
 *   - Skip semantics: spans whose resolved marker is `null` (or whose
 *     label isn't in `enabledSet`) are left in the output verbatim but
 *     still appear in the returned `applied` list (so callers can see
 *     what was detected but not redacted).
 *   - Per-span index is passed to function-form markers, counting only
 *     enabled spans — useful for re-insertable patterns like `[NAME_0]`.
 */

import type {
  DetectedSpan,
  MarkerStrategy,
  SpanLabel,
} from "../types.js";

export interface ApplyRedactionResult {
  redactedText: string;
  /**
   * The spans enriched with the marker string actually used (or the
   * default `[label]` marker for enabled-but-unstyled spans; or the
   * original `text` when `null` returned from the strategy).
   */
  applied: DetectedSpan[];
}

export function applyRedaction(
  input: string,
  spans: readonly DetectedSpan[],
  enabledSet: ReadonlySet<SpanLabel>,
  strategy: MarkerStrategy | undefined,
): ApplyRedactionResult {
  const sorted = [...spans].sort((a, b) => a.start - b.start);

  // A span is "enabled for redaction" when:
  //   - It's a rule span (custom rules apply unconditionally; the
  //     caller authored them, so the model's enabledCategories filter
  //     doesn't constrain them), OR
  //   - It's a model span whose label is in enabledSet.
  const isEnabled = (span: DetectedSpan): boolean =>
    span.source === "rule" ||
    enabledSet.has(span.label as SpanLabel);

  // First pass: resolve markers. Build a decorated array in original
  // order so we can index-count "enabled" spans consistently for
  // function-form markers.
  const decorated: DetectedSpan[] = [];
  let enabledIndex = 0;
  for (const span of sorted) {
    if (!isEnabled(span)) {
      decorated.push({ ...span, marker: span.text });
      continue;
    }
    // Rule spans carry their own preferred marker on the span object
    // (set by `runRules`). Caller-supplied marker strategy still wins
    // when it explicitly maps the rule's label to something non-null.
    const resolved = resolveMarker(span, enabledIndex, strategy);
    enabledIndex++;
    decorated.push({ ...span, marker: resolved ?? span.text });
  }

  // Second pass: splice from last to first so earlier offsets don't shift.
  let redacted = input;
  for (let i = decorated.length - 1; i >= 0; i--) {
    const span = decorated[i]!;
    if (!isEnabled(span)) continue;
    if (span.marker === span.text) continue; // strategy returned null or unchanged
    redacted = redacted.slice(0, span.start) + span.marker + redacted.slice(span.end);
  }

  return { redactedText: redacted, applied: decorated };
}

function resolveMarker(
  span: DetectedSpan,
  enabledIndex: number,
  strategy: MarkerStrategy | undefined,
): string | null {
  // Rule spans carry their explicit marker on the span itself (set by
  // `runRules` from `Rule.marker`). Use it as the default when the
  // caller hasn't passed a strategy.
  const ruleMarker = span.source === "rule" ? span.marker : `[${span.label}]`;
  if (strategy === undefined) {
    return ruleMarker;
  }
  if (typeof strategy === "function") {
    return strategy(span, enabledIndex);
  }
  const override = (strategy as Record<string, string | null | undefined>)[span.label];
  if (override === null) return null;
  if (override === undefined) return ruleMarker;
  return override;
}
