/**
 * Tabular-data helpers — `classifyColumns` and `redactTable`. Built
 * on top of the existing per-cell `detect()` / `redact()` so the
 * detection logic is identical to the text path; only the orchestration
 * is new (sampling + per-column aggregation).
 *
 * Use case: you have a CSV / DB dump / spreadsheet and want to know
 * which columns contain PII, or want to ship a redacted copy without
 * eyeballing every column. AWS Macie / BigID charge enterprise prices
 * for this; this is the local-first OSS version.
 */

import type {
  ColumnClassification,
  ClassifyTableOptions,
  DetectResult,
  DetectedSpan,
  MarkerStrategy,
  RedactOptions,
  RedactResult,
  RedactTableMode,
  RedactTableOptions,
  SpanLabel,
} from "../types.js";
import { markerPresets } from "./marker-presets.js";

const DEFAULT_SAMPLE_SIZE = 50;

interface DetectFn {
  (text: string, opts?: RedactOptions): Promise<DetectResult>;
}
interface RedactFn {
  (text: string, opts?: RedactOptions): Promise<RedactResult>;
}

/**
 * Resolve which row index data starts at and the per-column header.
 * Header sources (in priority order):
 *   1. `headers` option — explicit, takes precedence.
 *   2. `headerRow: true` — first row of `rows` is treated as headers.
 *   3. neither — no headers, classifications come back without `header`.
 */
function resolveHeaders(
  rows: readonly (readonly string[])[],
  opts: ClassifyTableOptions,
): { dataStart: number; headers: readonly (string | undefined)[] } {
  const ncols = rows[0]?.length ?? 0;
  if (opts.headers) {
    if (opts.headers.length !== ncols) {
      throw new Error(
        `headers length (${opts.headers.length}) does not match column count (${ncols})`,
      );
    }
    return { dataStart: 0, headers: opts.headers };
  }
  if (opts.headerRow) {
    return {
      dataStart: 1,
      headers: rows[0] ?? [],
    };
  }
  return { dataStart: 0, headers: new Array(ncols).fill(undefined) };
}

/**
 * Pick the most-frequent label from a count distribution. Ties broken
 * by insertion order (whichever appeared first in the cells). Returns
 * `null` if the distribution is empty.
 */
function pickMajorityLabel(
  labelCounts: Map<string, number>,
): { label: string | null; count: number } {
  let best: string | null = null;
  let bestCount = 0;
  for (const [label, count] of labelCounts) {
    if (count > bestCount) {
      best = label;
      bestCount = count;
    }
  }
  return { label: best, count: bestCount };
}

/**
 * Classify each column of `rows` by sampling up to `sampleSize` cells
 * per column and running the model + rule engine on each sample. The
 * column's label is the most-frequent label across samples; confidence
 * is the fraction of samples that matched.
 */
export async function classifyColumns(
  detect: DetectFn,
  rows: readonly (readonly string[])[],
  opts: ClassifyTableOptions = {},
): Promise<ColumnClassification[]> {
  if (rows.length === 0) return [];
  const { dataStart, headers } = resolveHeaders(rows, opts);
  const ncols = rows[0]?.length ?? 0;
  if (ncols === 0) return [];

  const sampleSize = opts.sampleSize ?? DEFAULT_SAMPLE_SIZE;
  const detectOpts: RedactOptions = {
    rules: opts.rules,
    presets: opts.presets,
    signal: opts.signal,
  };

  const out: ColumnClassification[] = [];

  for (let col = 0; col < ncols; col++) {
    const labelCounts = new Map<string, number>();
    let samples = 0;

    for (let r = dataStart; r < rows.length && samples < sampleSize; r++) {
      const cell = rows[r]?.[col];
      if (cell === undefined || cell === null) continue;
      const trimmed = cell.trim();
      if (trimmed.length === 0) continue;
      samples++;

      const result = await detect(cell, detectOpts);
      // Each cell can have multiple spans (e.g. "Hi Alice, my email is
      // alice@example.com" → person + email). For column-level
      // classification we care about which labels appeared at all in
      // this cell — count each label at most once per cell.
      const labelsInCell = new Set<string>();
      for (const span of result.spans) labelsInCell.add(span.label);
      for (const label of labelsInCell) {
        labelCounts.set(label, (labelCounts.get(label) ?? 0) + 1);
      }
    }

    const { label, count } = pickMajorityLabel(labelCounts);
    out.push({
      index: col,
      ...(headers[col] !== undefined ? { header: headers[col]! } : {}),
      label: label as SpanLabel | string | null,
      confidence: samples === 0 ? 0 : count / samples,
      samples,
      labelCounts: Object.freeze(Object.fromEntries(labelCounts)),
    });
  }

  return out;
}

/**
 * Redact / synth / drop columns of `rows` based on per-column
 * classifications. If `opts.classifications` is supplied, skip the
 * classify step and use them directly.
 */
export async function redactTable(
  detect: DetectFn,
  redact: RedactFn,
  baseMarkers: MarkerStrategy | undefined,
  rows: readonly (readonly string[])[],
  opts: RedactTableOptions = {},
): Promise<string[][]> {
  if (rows.length === 0) return [];
  const ncols = rows[0]?.length ?? 0;
  if (ncols === 0) return rows.map(() => []);

  const mode: RedactTableMode = opts.mode ?? "redact";
  const classifications =
    opts.classifications ?? (await classifyColumns(detect, rows, opts));

  // For "synth" mode: build a per-call faker if the filter's own
  // markers aren't already a faker. We use one faker for the whole
  // table so consistency holds across rows.
  const synthMarkers: MarkerStrategy | undefined =
    mode === "synth" ? markerPresets.faker() : undefined;
  const cellMarkers = synthMarkers ?? opts.markers ?? baseMarkers;

  const piiCols = new Set<number>();
  for (const c of classifications) if (c.label !== null) piiCols.add(c.index);

  const dropCols = mode === "drop_column" ? piiCols : new Set<number>();

  const { dataStart, headers } = resolveHeaders(rows, opts);
  const out: string[][] = [];

  // If headerRow was set, emit a header row first (filtered for dropped cols).
  if (opts.headerRow && headers.length > 0) {
    const headerRow: string[] = [];
    for (let c = 0; c < ncols; c++) {
      if (dropCols.has(c)) continue;
      headerRow.push(headers[c] ?? "");
    }
    out.push(headerRow);
  } else if (opts.headers) {
    // Explicit headers were passed but the input rows don't include
    // a header row — caller manages headers separately, don't emit.
  }

  for (let r = dataStart; r < rows.length; r++) {
    const inRow = rows[r] ?? [];
    const outRow: string[] = [];
    for (let c = 0; c < ncols; c++) {
      if (dropCols.has(c)) continue;
      const cell = inRow[c] ?? "";
      if (!piiCols.has(c) || cell.trim().length === 0) {
        outRow.push(cell);
        continue;
      }
      // PII column → run redact on the cell content. Per-cell detect
      // gives correct results for cells that contain more than just
      // the PII (e.g. "Email me at alice@example.com please").
      const result = await redact(cell, {
        markers: cellMarkers,
        enabledCategories: opts.enabledCategories,
        rules: opts.rules,
        presets: opts.presets,
        signal: opts.signal,
      });
      outRow.push(result.redactedText);
    }
    out.push(outRow);
  }
  return out;
}

// Re-export the supporting types for ergonomics.
export type {
  ColumnClassification,
  ClassifyTableOptions,
  RedactTableMode,
  RedactTableOptions,
} from "../types.js";

// Re-exported for tests / docs that want to assert against `DetectedSpan`.
export type { DetectedSpan };
