// Playground for `markerPresets.faker()`. Loads the same model the
// main /playground/ uses, but pre-wires the faker MarkerStrategy so
// the output is realistic-looking fakes instead of [private_email]
// markers. Side-by-side rendering + a mappings table that highlights
// repeated original→fake pairs (the consistency feature).
import {
  PrivacyFilter,
  markerPresets,
  getCachedModelInfo,
} from "textsift/browser";

const $preset   = document.getElementById("ts-preset") as HTMLSelectElement;
const $input    = document.getElementById("ts-input") as HTMLTextAreaElement;
const $run      = document.getElementById("ts-run") as HTMLButtonElement;
const $status   = document.getElementById("ts-status") as HTMLElement;
const $orig     = document.getElementById("ts-original") as HTMLElement;
const $faked    = document.getElementById("ts-faked") as HTMLElement;
const $mappings = document.getElementById("ts-mappings") as HTMLElement;
const $code     = document.getElementById("ts-code") as HTMLElement;
const $download = document.getElementById("ts-download-warning") as HTMLElement;

const presetsScript = document.getElementById("ts-presets") as HTMLScriptElement;
const PRESETS: Record<string, string> = JSON.parse(presetsScript.textContent ?? "{}");

let filterPromise: Promise<PrivacyFilter> | null = null;
let currentFilter: PrivacyFilter | null = null;
// One faker instance per filter, recreated on reload, so consistency
// state matches the filter's lifetime.
let faker = markerPresets.faker();

function setStatus(msg: string): void { $status.textContent = msg; }

function escape(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

$preset.addEventListener("change", () => {
  if ($preset.value === "custom") return; // leave whatever the user typed
  $input.value = PRESETS[$preset.value] ?? "";
});

async function confirmDownloadIfNeeded(): Promise<boolean> {
  if (filterPromise) return true;
  let info;
  try { info = await getCachedModelInfo(); }
  catch { return true; }
  if (info.supported && info.entries.length > 0) return true;
  const msg = info.supported
    ? "First run will download ~770 MB of model weights from HuggingFace and cache them on this device (OPFS). Continue?"
    : "OPFS is not available in this browser, so the ~770 MB model weights will be re-downloaded on every visit. Continue?";
  return window.confirm(msg);
}

async function getFilter(): Promise<PrivacyFilter> {
  if (filterPromise) return filterPromise;
  setStatus("loading model (first run downloads ~770 MB; cached after)…");
  filterPromise = PrivacyFilter.create({
    backend: "auto",
    markers: faker,
  })
    .then((f) => {
      currentFilter = f;
      setStatus("model ready");
      $download.hidden = true;
      return f;
    })
    .catch((err: Error) => {
      filterPromise = null;
      setStatus("load failed: " + err.message);
      throw err;
    });
  return filterPromise;
}

interface SpanLike {
  start: number;
  end: number;
  text: string;
  marker: string;
  label: string;
}

function renderSidebySide(input: string, spans: SpanLike[]): void {
  if (!spans.length) {
    $orig.textContent = input;
    $faked.textContent = input;
    return;
  }
  const sorted = [...spans].sort((a, b) => a.start - b.start);

  let origHtml = "", fakedHtml = "", cursor = 0, fakedCursor = 0;
  let fakedText = ""; // build the full faked string for cursor accounting

  for (const s of sorted) {
    origHtml += escape(input.slice(cursor, s.start));
    origHtml += `<mark class="ts-fake" title="${escape(s.label)}">${escape(s.text)}</mark>`;
    fakedText += input.slice(cursor, s.start) + s.marker;
    cursor = s.end;
  }
  origHtml += escape(input.slice(cursor));
  fakedText += input.slice(cursor);

  // Now build the faked side with marks around each replacement.
  cursor = 0; fakedCursor = 0;
  for (const s of sorted) {
    const before = input.slice(cursor, s.start);
    fakedHtml += escape(before);
    fakedCursor += before.length;
    fakedHtml += `<mark class="ts-fake" title="${escape(s.label)} (was: ${escape(s.text)})">${escape(s.marker)}</mark>`;
    fakedCursor += s.marker.length;
    cursor = s.end;
  }
  fakedHtml += escape(input.slice(cursor));

  $orig.innerHTML = origHtml;
  $faked.innerHTML = fakedHtml;
}

function renderMappings(spans: SpanLike[]): void {
  if (!spans.length) {
    $mappings.innerHTML = `<em class="ts-hint">No PII detected — nothing to fake.</em>`;
    return;
  }
  // Build mapping rows; mark "repeat" rows where the same original text
  // appears more than once and gets the same fake.
  const seen = new Map<string, number>();
  for (const s of spans) {
    const key = `${s.label}|${s.text}`;
    seen.set(key, (seen.get(key) ?? 0) + 1);
  }
  const sorted = [...spans].sort((a, b) => a.start - b.start);
  const rows = sorted.map((s) => {
    const key = `${s.label}|${s.text}`;
    const isRepeat = (seen.get(key) ?? 0) > 1;
    return `<tr class="${isRepeat ? "ts-repeat" : ""}">
      <td class="ts-orig">${escape(s.text)}</td>
      <td class="ts-fake-cell">${escape(s.marker)}</td>
      <td class="ts-label">${escape(s.label)}</td>
    </tr>`;
  }).join("");
  $mappings.innerHTML = `
    <table>
      <thead><tr><th>Original</th><th>Fake</th><th>Label</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function renderCode(): void {
  $code.textContent = [
    'import { PrivacyFilter, markerPresets } from "textsift";',
    "",
    "const filter = await PrivacyFilter.create({",
    "  markers: markerPresets.faker(),",
    "});",
    "",
    "const { redactedText } = await filter.redact(input);",
    "// Same input text → same fake within the filter's lifetime",
  ].join("\n");
}

$run.addEventListener("click", async () => {
  $run.disabled = true;
  try {
    if (!(await confirmDownloadIfNeeded())) {
      setStatus("cancelled — model not downloaded");
      return;
    }
    const text = $input.value;
    const filter = await getFilter();
    setStatus("running…");
    const result = await filter.redact(text);
    renderSidebySide(text, result.spans as SpanLike[]);
    renderMappings(result.spans as SpanLike[]);
    renderCode();
    setStatus(`${result.spans.length} span(s) faked`);
  } catch (err) {
    setStatus("error: " + (err as Error).message);
  } finally {
    $run.disabled = false;
  }
});

// Initial render so the page isn't empty.
renderCode();
(async () => {
  try {
    const info = await getCachedModelInfo();
    if (!info.supported || info.entries.length === 0) $download.hidden = false;
  } catch { /* not critical */ }
})();
