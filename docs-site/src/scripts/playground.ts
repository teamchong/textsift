  // Browser entry of the textsift package — pure WASM/WebGPU, no
  // native binary. Bundlers resolve `textsift/browser` and never
  // touch the native NAPI entry point reserved for Node.
  import { PrivacyFilter, getCachedModelInfo, clearCachedModel } from "textsift/browser";

  const ALL_CATEGORIES = [
    "private_person", "private_email", "private_phone", "private_address",
    "private_url", "private_date", "account_number", "secret",
  ];

  const $input  = document.getElementById("ts-input");
  const $run    = document.getElementById("ts-run");
  const $redact = document.getElementById("ts-redact");
  const $status = document.getElementById("ts-status");
  const $output = document.getElementById("ts-output");
  const $spans  = document.getElementById("ts-spans");
  const $code   = document.getElementById("ts-code");

  const $backend = document.getElementById("ts-backend");
  const $modelSrc = document.getElementById("ts-model-source");
  const $markers = document.getElementById("ts-markers");
  const $reload = document.getElementById("ts-reload");
  const $configHint = document.getElementById("ts-config-hint");

  const $refreshStorage = document.getElementById("ts-refresh-storage");
  const $wipe = document.getElementById("ts-wipe");
  const $storageStatus = document.getElementById("ts-storage-status");
  const $storageInfo = document.getElementById("ts-storage-info");
  const $downloadWarning = document.getElementById("ts-download-warning");

  let filterPromise = null;
  let currentFilter = null;

  // The library's default modelSource. The input field is pre-filled
  // with this so users can see the expected shape; we only pass it
  // through to PrivacyFilter when it differs from the default (the
  // library uses the same default internally).
  const DEFAULT_MODEL_SOURCE = "https://huggingface.co/openai/privacy-filter";

  function readConfig() {
    const backend = $backend.value;
    const sourceRaw = $modelSrc.value.trim();
    const modelSource = (sourceRaw && sourceRaw !== DEFAULT_MODEL_SOURCE) ? sourceRaw : undefined;
    const enabledCategories = ALL_CATEGORIES.filter(
      (c) => document.querySelector(`input[type="checkbox"][data-cat="${c}"]`).checked,
    );
    let markers;
    const raw = $markers.value.trim();
    if (raw) {
      try { markers = JSON.parse(raw); }
      catch (err) { throw new Error("Invalid markers JSON: " + err.message); }
    }
    const opts = { backend };
    if (modelSource) opts.modelSource = modelSource;
    // enabledCategories goes to the per-call detect/redact options;
    // markers can be set on either create() or per-call. Keep them
    // per-call here so toggling a checkbox doesn't force a model
    // reload.
    return { createOpts: opts, callOpts: { enabledCategories, markers } };
  }

  function buildCodeSnippet(createOpts, callOpts, mode) {
    const { backend, modelSource } = createOpts;
    const createPretty = JSON.stringify(
      { backend, ...(modelSource ? { modelSource } : {}) },
      null, 2,
    );
    const callPretty = JSON.stringify(
      Object.fromEntries(
        Object.entries(callOpts).filter(
          ([, v]) => v !== undefined && !(Array.isArray(v) && v.length === ALL_CATEGORIES.length),
        ),
      ),
      null, 2,
    );
    const callArgs = callPretty === "{}" ? "input" : `input, ${callPretty}`;
    return [
      'import { PrivacyFilter } from "textsift/browser";',
      "",
      `const filter = await PrivacyFilter.create(${createPretty});`,
      `const result = await filter.${mode}(${callArgs});`,
      "filter.dispose();",
    ].join("\n");
  }

  function setStatus(msg) { $status.textContent = msg; }
  function setStorageStatus(msg) { $storageStatus.textContent = msg; }

  function escape(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function renderSpans(text, spans) {
    if (!spans.length) return escape(text);
    const sorted = [...spans].sort((a, b) => a.start - b.start);
    let html = "", cursor = 0;
    for (const s of sorted) {
      html += escape(text.slice(cursor, s.start));
      html += `<mark class="ts-span" data-label="${escape(s.label)}" title="${escape(s.label)}">${escape(text.slice(s.start, s.end))}</mark>`;
      cursor = s.end;
    }
    html += escape(text.slice(cursor));
    return html;
  }

  function renderRedacted(text, spans) {
    if (!spans.length) return escape(text);
    const sorted = [...spans].sort((a, b) => a.start - b.start);
    let html = "", cursor = 0;
    for (const s of sorted) {
      const marker = s.marker ?? `[${s.label}]`;
      html += escape(text.slice(cursor, s.start));
      html += `<mark class="ts-span ts-marker" data-label="${escape(s.label)}" title="${escape(s.label)} (was: ${escape(s.text)})">${escape(marker)}</mark>`;
      cursor = s.end;
    }
    html += escape(text.slice(cursor));
    return html;
  }

  // Returns true if the user accepts the download (or no warning is needed).
  // Shows a confirm() once when nothing is cached so a click on Detect/Redact
  // doesn't silently start a ~770 MB fetch.
  async function confirmDownloadIfNeeded() {
    if (filterPromise) return true; // model already loaded this session
    let info;
    try { info = await getCachedModelInfo(); }
    catch { return true; } // can't check — let it proceed
    if (info.supported && info.entries.length > 0) return true;
    const msg = info.supported
      ? "First run will download ~770 MB of model weights from HuggingFace and " +
        "cache them on this device (OPFS). Continue?"
      : "OPFS is not available in this browser, so the ~770 MB model weights " +
        "will be re-downloaded on every visit. Continue?";
    return window.confirm(msg);
  }

  async function getFilter(createOpts) {
    if (filterPromise) return filterPromise;
    setStatus("loading model (first run downloads ~770 MB; cached after)…");
    filterPromise = PrivacyFilter.create(createOpts)
      .then((f) => {
        currentFilter = f;
        setStatus("model ready");
        // Cache is now populated — hide the warning banner.
        refreshStorage();
        return f;
      })
      .catch((err) => { filterPromise = null; setStatus("load failed: " + err.message); throw err; });
    return filterPromise;
  }

  async function disposeFilter() {
    if (currentFilter) {
      try { currentFilter.dispose(); } catch {}
      currentFilter = null;
    }
    filterPromise = null;
  }

  async function withButtonsDisabled(fn) {
    $run.disabled = true; $redact.disabled = true; $reload.disabled = true; $wipe.disabled = true;
    try { return await fn(); }
    finally { $run.disabled = false; $redact.disabled = false; $reload.disabled = false; $wipe.disabled = false; }
  }

  $reload.addEventListener("click", () => withButtonsDisabled(async () => {
    let cfg;
    try { cfg = readConfig(); }
    catch (err) { $configHint.textContent = err.message; return; }
    $configHint.textContent = "";
    // Reload always re-creates the filter; warn if cache is empty so
    // the user knows a fresh fetch is about to happen.
    if (!(await confirmDownloadIfNeeded())) {
      setStatus("cancelled — model not loaded");
      return;
    }
    setStatus("reloading…");
    await disposeFilter();
    await getFilter(cfg.createOpts);
  }));

  $run.addEventListener("click", () => withButtonsDisabled(async () => {
    let cfg;
    try { cfg = readConfig(); }
    catch (err) { setStatus(err.message); return; }
    if (!(await confirmDownloadIfNeeded())) {
      setStatus("cancelled — model not downloaded");
      return;
    }
    const text = $input.value;
    const filter = await getFilter(cfg.createOpts);
    setStatus("detecting…");
    const result = await filter.detect(text, cfg.callOpts);
    $output.innerHTML = renderSpans(text, result.spans);
    $spans.textContent = JSON.stringify(result, null, 2);
    $code.textContent = buildCodeSnippet(cfg.createOpts, cfg.callOpts, "detect");
    setStatus(`${result.spans.length} span(s) detected`);
  }));

  $redact.addEventListener("click", () => withButtonsDisabled(async () => {
    let cfg;
    try { cfg = readConfig(); }
    catch (err) { setStatus(err.message); return; }
    if (!(await confirmDownloadIfNeeded())) {
      setStatus("cancelled — model not downloaded");
      return;
    }
    const text = $input.value;
    const filter = await getFilter(cfg.createOpts);
    setStatus("redacting…");
    const result = await filter.redact(text, cfg.callOpts);
    $output.innerHTML = renderRedacted(text, result.spans);
    $spans.textContent = JSON.stringify(result, null, 2);
    $code.textContent = buildCodeSnippet(cfg.createOpts, cfg.callOpts, "redact");
    setStatus(`${result.spans.length} span(s) replaced`);
  }));

  // ── Storage panel ───────────────────────────────────────

  function fmtBytes(n) {
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
    return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
  }

  async function refreshStorage() {
    setStorageStatus("checking…");
    try {
      const info = await getCachedModelInfo();
      const cacheEmpty = !info.supported || info.entries.length === 0;
      $downloadWarning.hidden = !cacheEmpty;
      if (!info.supported) {
        $storageInfo.innerHTML = "<em>OPFS not available in this browser. Weights re-download every visit.</em>";
        setStorageStatus("");
        return;
      }
      if (info.entries.length === 0) {
        $storageInfo.innerHTML = "<em>Nothing cached yet. The first Detect or Redact will download ~770 MB.</em>";
        setStorageStatus("0 bytes");
        return;
      }
      const rows = info.entries
        .sort((a, b) => b.bytes - a.bytes)
        .map((e) => `<tr><td>${escape(e.name)}</td><td>${fmtBytes(e.bytes)}</td></tr>`)
        .join("");
      $storageInfo.innerHTML = `
        <table>
          <thead><tr><th>file</th><th>size</th></tr></thead>
          <tbody>${rows}</tbody>
          <tfoot><tr class="ts-total"><td>total (${info.entries.length} files)</td><td>${fmtBytes(info.totalBytes)}</td></tr></tfoot>
        </table>
      `;
      setStorageStatus(fmtBytes(info.totalBytes));
    } catch (err) {
      $storageInfo.innerHTML = `<em>error: ${escape(String(err))}</em>`;
      setStorageStatus("");
    }
  }

  $refreshStorage.addEventListener("click", () => refreshStorage());

  $wipe.addEventListener("click", () => withButtonsDisabled(async () => {
    if (!confirm("Wipe all cached textsift weights from this device? Next Detect / Redact will re-download ~770 MB.")) return;
    setStorageStatus("wiping…");
    await disposeFilter();
    const { removed, bytes } = await clearCachedModel();
    setStorageStatus(`wiped ${removed} files (${fmtBytes(bytes)})`);
    await refreshStorage();
    setStatus("idle (model unloaded)");
  }));

  // Initial render
  refreshStorage();
  $code.textContent = buildCodeSnippet({ backend: "auto" }, {}, "detect");
