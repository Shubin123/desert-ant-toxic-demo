// SentencePiece-Unigram tokenizer implemented from the model's tokenizer.json
// (XLM-R lineage, trimmed to the model's eight languages, 60,728-piece
// vocabulary). Mirrors the
// HuggingFace `tokenizers` pipeline the training/eval code used:
//
//   WhitespaceSplit -> Metaspace("▁", prepend always) -> Unigram Viterbi
//   -> <s> ... </s> template -> truncate/pad to the model's fixed window.
//
// The upstream pipeline also runs a `Precompiled` (sentencepiece nmt_nfkc)
// normalizer; the demo feeds text that already went through
// `normalize()` (NFKC + lowercase), where the charsmap is a no-op on
// everything the parity corpus covers. Special-token strings in user input
// (e.g. "<s>") are NOT parsed as control tokens, on purpose.
//
// Parity with the Python reference encoding is locked by
// tests/tokenizer.parity.test.js against fixtures generated with
// `transformers` from the same tokenizer.json.

const METASPACE = "▁";
const UNK_PENALTY = 10.0;

export class Tokenizer {
  constructor(tokenizerJSON) {
    const model = tokenizerJSON.model;
    if (model.type !== "Unigram") {
      throw new Error(`expected a Unigram tokenizer, got ${model.type}`);
    }
    this.vocab = new Map();
    let minScore = 0;
    let maxLen = 1;
    for (let id = 0; id < model.vocab.length; id++) {
      const [piece, score] = model.vocab[id];
      this.vocab.set(piece, { id, score });
      if (score < minScore) minScore = score;
      const cp = Array.from(piece).length;
      if (cp > maxLen) maxLen = cp;
    }
    this.maxPieceLen = maxLen;
    this.unkId = model.unk_id;
    this.unkScore = minScore - UNK_PENALTY;
    const specials = Object.fromEntries(
      (tokenizerJSON.added_tokens ?? []).map((t) => [t.content, t.id]));
    this.bosId = specials["<s>"] ?? 0;
    this.eosId = specials["</s>"] ?? 2;
    this.padId = specials["<pad>"] ?? 1;
  }

  /** Unigram Viterbi over one whitespace-free chunk (code points). */
  tokenizeChunk(chunk) {
    const cps = Array.from(METASPACE + chunk);
    const n = cps.length;
    // best score to reach boundary i, and the piece that got there
    const best = new Float64Array(n + 1).fill(-Infinity);
    const backPiece = new Array(n + 1).fill(null);
    const backFrom = new Int32Array(n + 1);
    best[0] = 0;
    for (let i = 1; i <= n; i++) {
      const maxL = Math.min(this.maxPieceLen, i);
      for (let len = 1; len <= maxL; len++) {
        const from = i - len;
        if (best[from] === -Infinity) continue;
        const piece = cps.slice(from, i).join("");
        const entry = this.vocab.get(piece);
        if (entry) {
          const s = best[from] + entry.score;
          if (s > best[i]) {
            best[i] = s;
            backPiece[i] = entry.id;
            backFrom[i] = from;
          }
        } else if (len === 1) {
          // unknown single code point
          const s = best[from] + this.unkScore;
          if (s > best[i]) {
            best[i] = s;
            backPiece[i] = this.unkId;
            backFrom[i] = from;
          }
        }
      }
    }
    const ids = [];
    for (let i = n; i > 0; i = backFrom[i]) ids.push(backPiece[i]);
    ids.reverse();
    // fuse consecutive <unk> (tokenizers' fuse_unk, on for converted
    // sentencepiece models)
    const fused = [];
    for (const id of ids) {
      if (id === this.unkId && fused[fused.length - 1] === this.unkId) continue;
      fused.push(id);
    }
    return fused;
  }

  /** Piece ids for already-normalized text, no specials, no padding. */
  tokenize(text) {
    const ids = [];
    for (const chunk of text.split(/\s+/u)) {
      if (chunk) ids.push(...this.tokenizeChunk(chunk));
    }
    return ids;
  }

  /**
   * Full model encoding: `<s> ids </s>` truncated and padded to maxLength.
   * Returns { inputIds, attentionMask } as plain arrays.
   */
  encode(text, maxLength = 128) {
    let ids = this.tokenize(text);
    if (ids.length > maxLength - 2) ids = ids.slice(0, maxLength - 2);
    const inputIds = [this.bosId, ...ids, this.eosId];
    const attentionMask = new Array(inputIds.length).fill(1);
    while (inputIds.length < maxLength) {
      inputIds.push(this.padId);
      attentionMask.push(0);
    }
    return { inputIds, attentionMask };
  }
}

