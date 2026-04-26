// Shared synthetic-weights config for the native forward benches.
// Mirrors openai/privacy-filter's model dimensions exactly so the
// per-kernel work and weight upload size match production.
export const PF = Object.freeze({
  hiddenSize: 640,
  numHeads: 14,
  numKvHeads: 2,
  headDim: 64,
  slidingWindow: 128,
  intermediateSize: 640,
  numExpertsPerTok: 4,
  numExperts: 128,
  rmsNormEps: 1e-5,
  numLayers: 8,
  numClasses: 33,
  vocabSize: 200000,
});
