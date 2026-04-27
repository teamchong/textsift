/**
 * SARIF (Static Analysis Results Interchange Format) export. Lets
 * textsift findings flow into GitHub Code Scanning, GitLab SAST,
 * Sonar, etc. without each tool re-implementing a parser for our
 * native JSON shape.
 *
 * Spec: https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html
 *
 * We emit SARIF v2.1.0 with the minimum required structure plus the
 * fields downstream tools actually use:
 *   - `tool.driver.{name, version, informationUri, rules}` so each
 *     finding has a clickable rule definition
 *   - `results[].{ruleId, level, message, locations}` per finding
 *   - `level` mapped from severity: model+block → error, warn → warning
 *   - `partialFingerprints` so reruns don't duplicate findings in
 *     GitHub Code Scanning's deduplication
 */

import { createHash } from "node:crypto";
import type { DetectedSpan, DetectResult, SpanLabel } from "./browser/types.js";

const TOOL_VERSION = "0.1.0"; // bumped per release
const TOOL_INFO_URI = "https://teamchong.github.io/textsift/";

// Per-label SARIF rule definitions. The model has 8 labels; rule
// spans (from the secrets preset etc.) carry their own label string,
// which we map to a synthesised rule on demand.
const MODEL_RULES: Record<SpanLabel, { name: string; description: string }> = {
  private_email: { name: "PrivateEmail",   description: "Email address" },
  private_phone: { name: "PrivatePhone",   description: "Phone number" },
  private_person: { name: "PrivatePerson", description: "Personal name" },
  private_address: { name: "PrivateAddress", description: "Postal address" },
  private_url: { name: "PrivateUrl",       description: "Personal / sensitive URL (account recovery, session-id, etc.)" },
  private_date: { name: "PrivateDate",     description: "Date tied to a person (birth date, signup date, etc.)" },
  account_number: { name: "AccountNumber", description: "Account / card / SSN / customer ID" },
  secret: { name: "Secret",                description: "Password / API key / JWT / recovery code" },
};

interface SarifLocation {
  physicalLocation: {
    artifactLocation: { uri: string };
    region: { startLine: number; startColumn: number; endLine: number; endColumn: number };
  };
}

interface SarifResult {
  ruleId: string;
  level: "error" | "warning" | "note" | "none";
  message: { text: string };
  locations: SarifLocation[];
  partialFingerprints?: { primaryLocationLineHash: string };
}

interface SarifRule {
  id: string;
  name: string;
  shortDescription: { text: string };
  defaultConfiguration: { level: "error" | "warning" | "note" };
  helpUri?: string;
}

interface SarifLog {
  $schema: string;
  version: "2.1.0";
  runs: Array<{
    tool: { driver: { name: string; version: string; informationUri: string; rules: SarifRule[] } };
    results: SarifResult[];
  }>;
}

function lineColOf(text: string, offset: number): { line: number; col: number } {
  let line = 1, col = 1;
  for (let i = 0; i < offset && i < text.length; i++) {
    if (text[i] === "\n") { line++; col = 1; } else { col++; }
  }
  return { line, col };
}

function sarifLevelOf(span: DetectedSpan): "error" | "warning" | "note" {
  if (span.source === "rule") {
    if (span.severity === "block") return "error";
    if (span.severity === "warn") return "warning";
    return "note";
  }
  // Model spans: error by default. Caller can downgrade via the
  // `levelOverride` option below.
  return "error";
}

function fingerprint(file: string, line: number, label: string, text: string): string {
  return createHash("sha256")
    .update(`${file}|${line}|${label}|${text}`)
    .digest("hex")
    .slice(0, 16);
}

/** Per-file finding bundle for `toSarif`. */
export interface SarifFileFindings {
  /** Repo-relative path (or whatever URI scheme the consumer expects). */
  uri: string;
  /** Original input text — needed to map char offsets to line/col. */
  text: string;
  /** Spans returned by `filter.detect(text)`. */
  spans: readonly DetectedSpan[];
}

export interface ToSarifOptions {
  /**
   * Override the level for model spans. SARIF "error" fails the
   * GitHub Code Scanning check; "warning" surfaces in the Security
   * tab without failing. Default: "error".
   */
  modelSpanLevel?: "error" | "warning" | "note";
}

/**
 * Convert a list of per-file findings to a single SARIF v2.1.0 log.
 * Suitable for `--sarif > findings.sarif` or `JSON.stringify` for
 * direct shipping to GitHub via `codeql-action/upload-sarif`.
 */
export function toSarif(
  files: readonly SarifFileFindings[],
  opts: ToSarifOptions = {},
): SarifLog {
  const usedRuleIds = new Set<string>();
  const results: SarifResult[] = [];

  for (const file of files) {
    for (const span of file.spans) {
      const ruleId = span.label;
      usedRuleIds.add(ruleId);
      const start = lineColOf(file.text, span.start);
      const end = lineColOf(file.text, span.end);
      const baseLevel = sarifLevelOf(span);
      const level = (span.source === "model" && opts.modelSpanLevel)
        ? opts.modelSpanLevel
        : baseLevel;
      const preview = span.text.length > 80 ? span.text.slice(0, 77) + "..." : span.text;
      results.push({
        ruleId,
        level,
        message: { text: `Found ${ruleId}: "${preview}"` },
        locations: [{
          physicalLocation: {
            artifactLocation: { uri: file.uri },
            region: {
              startLine: start.line,
              startColumn: start.col,
              endLine: end.line,
              endColumn: end.col,
            },
          },
        }],
        partialFingerprints: {
          primaryLocationLineHash: fingerprint(file.uri, start.line, ruleId, span.text),
        },
      });
    }
  }

  const rules: SarifRule[] = [];
  for (const id of usedRuleIds) {
    const modelDef = (MODEL_RULES as Record<string, typeof MODEL_RULES[SpanLabel]>)[id];
    if (modelDef) {
      rules.push({
        id,
        name: modelDef.name,
        shortDescription: { text: modelDef.description },
        defaultConfiguration: { level: opts.modelSpanLevel ?? "error" },
        helpUri: `${TOOL_INFO_URI}backends/`,
      });
    } else {
      // Rule span (from a custom rule or the secrets preset). The
      // label string IS the rule id; we synthesise minimal metadata.
      rules.push({
        id,
        name: id,
        shortDescription: { text: `textsift custom rule: ${id}` },
        defaultConfiguration: { level: "error" },
        helpUri: `${TOOL_INFO_URI}api/`,
      });
    }
  }

  return {
    $schema: "https://json.schemastore.org/sarif-2.1.0.json",
    version: "2.1.0",
    runs: [{
      tool: {
        driver: {
          name: "textsift",
          version: TOOL_VERSION,
          informationUri: TOOL_INFO_URI,
          rules,
        },
      },
      results,
    }],
  };
}

/**
 * Convenience: convert a single `DetectResult` for one file. Use
 * `toSarif(...)` directly when scanning multiple files in one run.
 */
export function detectResultToSarif(
  result: DetectResult,
  filePath: string,
  opts: ToSarifOptions = {},
): SarifLog {
  return toSarif(
    [{ uri: filePath, text: result.input, spans: result.spans }],
    opts,
  );
}
