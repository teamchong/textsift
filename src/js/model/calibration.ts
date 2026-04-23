/**
 * Parse the Viterbi-calibration JSON artifact that ships alongside the
 * privacy-filter weights.
 *
 * Schema (matches upstream `opf._core.decoding.VITERBI_BIAS_KEYS`):
 *   {
 *     "operating_points": {
 *       "default": {
 *         "biases": {
 *           "transition_bias_background_stay":      number,
 *           "transition_bias_background_to_start":  number,
 *           "transition_bias_inside_to_continue":   number,
 *           "transition_bias_inside_to_end":        number,
 *           "transition_bias_end_to_background":    number,
 *           "transition_bias_end_to_start":         number,
 *         }
 *       }
 *     }
 *   }
 *
 * The six biases parameterise the allowed BIOES transitions. They add to
 * the per-token emission scores inside the Viterbi search and calibrate
 * the precision/recall tradeoff — the upstream model ships a "default"
 * operating point tuned for balanced F1.
 */

import { PrivacyFilterError } from "../types.js";

export interface ViterbiCalibration {
  readonly backgroundStay: number;
  readonly backgroundToStart: number;
  readonly insideToContinue: number;
  readonly insideToEnd: number;
  readonly endToBackground: number;
  readonly endToStart: number;
}

const EXPECTED_BIAS_KEYS = [
  "transition_bias_background_stay",
  "transition_bias_background_to_start",
  "transition_bias_inside_to_continue",
  "transition_bias_inside_to_end",
  "transition_bias_end_to_background",
  "transition_bias_end_to_start",
] as const;

export function loadCalibration(json: unknown): ViterbiCalibration {
  const envelope = asObject(json, "calibration JSON");
  const opPoints = asObject(envelope.operating_points, "operating_points");
  const defaultPoint = asObject(opPoints.default, "operating_points.default");
  const biases = asObject(defaultPoint.biases, "operating_points.default.biases");

  for (const key of EXPECTED_BIAS_KEYS) {
    if (!(key in biases)) {
      throw new PrivacyFilterError(
        `calibration artifact missing required key: ${key}`,
        "CALIBRATION_INVALID",
      );
    }
    if (typeof biases[key] !== "number") {
      throw new PrivacyFilterError(
        `calibration artifact value for ${key} must be a number; got ${typeof biases[key]}`,
        "CALIBRATION_INVALID",
      );
    }
  }

  return Object.freeze({
    backgroundStay: biases.transition_bias_background_stay as number,
    backgroundToStart: biases.transition_bias_background_to_start as number,
    insideToContinue: biases.transition_bias_inside_to_continue as number,
    insideToEnd: biases.transition_bias_inside_to_end as number,
    endToBackground: biases.transition_bias_end_to_background as number,
    endToStart: biases.transition_bias_end_to_start as number,
  });
}

/** Default all-zero calibration — used for tests before a real artifact is loaded. */
export function zeroCalibration(): ViterbiCalibration {
  return Object.freeze({
    backgroundStay: 0,
    backgroundToStart: 0,
    insideToContinue: 0,
    insideToEnd: 0,
    endToBackground: 0,
    endToStart: 0,
  });
}

function asObject(value: unknown, field: string): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new PrivacyFilterError(
      `calibration artifact: expected object at ${field}, got ${describe(value)}`,
      "CALIBRATION_INVALID",
    );
  }
  return value as Record<string, unknown>;
}

function describe(value: unknown): string {
  if (value === null) return "null";
  if (Array.isArray(value)) return "array";
  return typeof value;
}
