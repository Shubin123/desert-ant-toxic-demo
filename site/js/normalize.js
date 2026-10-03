// JS port of toxic-training's `lexicon.normalize` (src/toxic_training/lexicon.py).
// The model was trained and evaluated on text canonicalized by this exact
// function, so the demo must apply it before tokenizing: NFKC, strip
// zero-width characters, homoglyph/leet to ASCII, collapse character repeats
// and single-character gaps, lowercase. Parity with the Python reference is
// locked by tests/tokenizer.parity.test.js.

const LEET = {
  "@": "a", "4": "a", "3": "e", "1": "i", "0": "o", "$": "s", "5": "s", "7": "t",
};

const ZERO_WIDTH = /[​‌‍⁠﻿]/g;

// Python `(.)\1{2,}` : `.` excludes only \n.
const REPEATS = /([^\n])\1{2,}/gu;

// Python `\b(\w) (?=\w\b)` with unicode \w and \b. JS \w is ASCII-only, so
// spell out the word class and both boundaries with lookarounds.
const W = "[\\p{L}\\p{N}_]";
const SINGLE_CHAR_GAPS = new RegExp(
  `(?<!${W})(${W}) (?=${W}(?!${W}))`, "gu");

export function normalize(text) {
  let t = text.normalize("NFKC").replace(ZERO_WIDTH, "");
  t = Array.from(t, (c) => LEET[c.toLowerCase()] ?? c).join("");
  t = t.replace(REPEATS, "$1$1");
  t = t.replace(SINGLE_CHAR_GAPS, "$1");
  return t.toLowerCase();
}

