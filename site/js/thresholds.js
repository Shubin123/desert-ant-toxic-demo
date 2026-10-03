// Flag semantics, mirrored from toxic_meta.json's `flag_semantics`:
// flagged = ANY content head at or above its threshold. HATEFUL alone is not
// the deploy gate. Each head's entry in the meta's `content_thresholds` block
// is either a plain number or `{ default, per_language }` (the v0.1.0 shape);
// per-language beats default, and HATEFUL additionally falls back to the
// `recommended` per-language block for older metas. Anything unreadable falls
// back to 0.5. DOM-free so node tests can lock the decision logic against the
// parity fixtures.

const DEFAULT_HEAD_THRESHOLD = 0.5;

/** Per-head decision thresholds for one language. */
export function contentThresholds(meta, lang) {
  const perLang = meta.recommended?.per_language_thresholds ?? {};
  const hateful = perLang[lang] ?? meta.recommended?.threshold ?? 0.4;
  const declared = meta.content_thresholds ?? {};
  const labels = meta.content_labels ?? Object.keys(declared);
  const thresholds = {};
  for (const label of labels) {
    const value = declared[label];
    if (typeof value === "number") {
      thresholds[label] = value;
    } else if (typeof value === "object" && value !== null) {
      const candidate = value.per_language?.[lang] ?? value.default;
      thresholds[label] = typeof candidate === "number"
        ? candidate
        : label === "HATEFUL" ? hateful : DEFAULT_HEAD_THRESHOLD;
    } else {
      thresholds[label] = label === "HATEFUL" ? hateful : DEFAULT_HEAD_THRESHOLD;
    }
  }
  return thresholds;
}

/**
 * Applies the thresholds to per-head probabilities.
 * Returns { flagged, fired } where fired lists the head labels at or above
 * their threshold, in the heads' original order.
 */
export function decide(contentProbs, thresholds) {
  const fired = Object.entries(contentProbs)
    .filter(([label, p]) => p >= (thresholds[label] ?? DEFAULT_HEAD_THRESHOLD))
    .map(([label]) => label);
  return { flagged: fired.length > 0, fired };
}

