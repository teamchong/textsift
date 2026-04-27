# NotebookLM prompt for the textsift intro deck

Paste this into NotebookLM's slide-generation prompt. The source
document is `intro-source.md` in the same directory.

---

Generate a 7-slide technical architecture deck from the attached
source document. Audience is engineers, not buyers. The author is a
solo developer who built this as a personal learning project; the
voice should reflect that — honest about constraints, no product-
launch hype.

## Hard constraints (do not violate)

1. **Quote numbers only from the source.** Never invent specifics
   like "4-minute walkthrough", "10× speedup", "41/41 inputs". If a
   number isn't in the source document, do not put it on a slide.
2. **Name hardware on every performance number.** "25 ms" is wrong;
   "25 ms on Intel Iris Xe via Vulkan-direct" is right. The source
   names hardware on every number — preserve that.
3. **No rigged baselines.** Don't compare to a "naive O(N²)" version
   nobody would actually write. The source talks about real
   alternatives the audience would consider (ONNX Runtime CPU,
   transformers.js, Chrome's own WebGPU stack via Dawn). Use those.
4. **Get the kernel name right.** The model is `model_q4f16.onnx` —
   int4 weights, fp16 activations. The matmul kernel is
   `matmul_int4_fp16`, NOT `matmul_bf16` (an earlier version of this
   deck got this wrong; do not repeat).
5. **The backend count.** There are five backends (WebGPU, WASM,
   Metal-direct, Vulkan-direct, Dawn-direct), not two. Do not say
   "Two Backends, One API".
6. **Conformance number.** The ONNX-reference parity test reports
   10/10 inputs span-equivalent at v0.1.0 — quote that, not a made-up
   number. Cite it as "ONNX reference" (the same `model_q4f16.onnx`
   file textsift loads, decoded with the same Viterbi+biases), not
   "PyTorch reference".

## Slide structure

Each slide is one technical insight. Title is specific, not a
buzzword. Subtitle is 1–2 sentences explaining what the audience
should take away. Diagram description names actual components, not
generic shapes.

Suggested arc (you can compress to 6 or expand to 8 if the source
material naturally bends that way):

1. **Why client-side.** The privacy-and-latency cost of sending raw
   text to a remote PII filter, vs running the model on the user's
   device. Cite the model (`openai/privacy-filter`) and that it
   downloads once (770 MB), not "zero network".

2. **The model and the canonical decoder.** What `openai/privacy-filter`
   is (8-label token classifier, q4f16, ~770 MB), and why naive argmax
   isn't the canonical decoder — the model ships
   `viterbi_calibration.json` with six transition biases that enforce
   BIOES legality. textsift mirrors that calibration.

3. **Five backends, one API.** Same TypeScript surface
   (`PrivacyFilter`), backend chosen at runtime: WebGPU + WASM in
   browsers; Metal-direct / Vulkan-direct / Dawn-direct in Node;
   WASM CPU fallback when no GPU. Per-platform native binary picked
   via npm `optionalDependencies`.

4. **The tokenizer is its own win.** Pure-TypeScript o200k BPE,
   76 KB gzipped, vs `@huggingface/transformers` pulling in
   `tokenizers` + `protobuf` + a long tail. Audience: anyone who's
   tried to put HF in a browser bundle.

5. **The Linux story.** Vulkan-direct fills the non-NVIDIA Linux
   gap — Intel Iris Xe at T=32 measures ~25 ms via Vulkan-direct vs
   ~785 ms via ONNX Runtime Node CPU on the same machine (32×). At
   ~50% of theoretical memory-bandwidth ceiling. Real numbers, real
   hardware.

6. **What I learned.** The Windows port took 10 commits to make
   work — Node header fetch, MSVC env translation, MSVC ABI target,
   `SIZE_MAX` translate-c bug, UCRT linkage, `node.lib` import
   library, PowerShell stderr swallowing. Each fix was small; the
   slog was diagnosis.

7. **Try it.** npm install, playground URL, docs URL, source URL,
   model URL. From the source document's "Where it lives" section,
   verbatim.

## Tone

Read the source document's "Why this exists" and "What I learned"
sections — that's the voice. Personal, specific, "I did X because
Y", not "we are excited to announce". The audience is the kind of
engineer who would read a Cloudflare blog post end-to-end. Don't
talk down to them and don't oversell.

## What NOT to add

- No "blazingly fast", no "production-ready", no rocket emojis.
- No claim that exceeds what's in the source document.
- No O-notation comparisons against strawman baselines.
- No "join the community" / "we're hiring" / call-to-action slide.
- No "powered by NotebookLM" / NotebookLM watermark on author-facing
  slides (it's fine on the corner; just don't make it a slide).

Output the deck.
