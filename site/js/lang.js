// Tiny stopword + diacritic language guesser for the model's eight languages.
// It exists only to pick the per-language decision threshold from
// toxic_meta.json; it is a heuristic, so the page always offers a manual
// override. Scores each language by hits of high-frequency function words
// (weight 2) and language-specific characters (weight 1); falls back to "en".

const STOPWORDS = {
  en: ["the", "and", "is", "are", "you", "of", "to", "that", "it", "not",
    "this", "they", "should", "have", "was", "all", "be", "with", "my"],
  de: ["der", "die", "das", "und", "ist", "nicht", "ich", "sie", "ein",
    "eine", "sind", "haben", "mit", "auf", "für", "wir", "keinen", "alle"],
  fr: ["le", "la", "les", "et", "est", "pas", "je", "vous", "une", "des",
    "que", "pour", "sont", "trop", "c'est", "dans", "au", "ce", "ils"],
  nl: ["de", "het", "een", "en", "is", "niet", "ik", "ze", "zijn", "met",
    "naar", "voor", "allemaal", "moeten", "worden", "mijn", "ga", "wij"],
  pt: ["o", "a", "os", "as", "e", "é", "não", "uma", "que", "para", "com",
    "eles", "todos", "devem", "ser", "do", "da", "em", "foi"],
  es: ["el", "la", "los", "las", "y", "es", "no", "una", "que", "para",
    "con", "son", "todos", "deben", "ser", "del", "en", "por", "mi"],
  it: ["il", "la", "i", "le", "e", "è", "non", "una", "che", "per", "con",
    "sono", "tutti", "del", "della", "di", "questo", "loro", "più"],
  pl: ["i", "nie", "to", "się", "jest", "są", "na", "do", "z", "że", "ich",
    "wszyscy", "być", "powinni", "tak", "ale", "po", "co", "dla"],
};

const CHARS = {
  de: "äöüß", fr: "àâçéèêëîïôùûœ", pt: "ãõçáéêóú", es: "ñáéíóú¿¡",
  it: "àèéìòù", pl: "ąćęłńóśźż", nl: "", en: "",
};

export const LANGUAGES = Object.keys(STOPWORDS);

export function guessLanguage(text) {
  const words = text.toLowerCase().split(/[^\p{L}']+/u).filter(Boolean);
  if (words.length === 0) return "en";
  const scores = {};
  for (const lang of LANGUAGES) {
    let s = 0;
    const stop = new Set(STOPWORDS[lang]);
    for (const w of words) if (stop.has(w)) s += 2;
    for (const c of CHARS[lang]) {
      for (const ch of text.toLowerCase()) if (ch === c) s += 1;
    }
    scores[lang] = s;
  }
  let bestLang = "en";
  let bestScore = 0;
  for (const lang of LANGUAGES) {
    if (scores[lang] > bestScore) {
      bestScore = scores[lang];
      bestLang = lang;
    }
  }
  return bestScore > 0 ? bestLang : "en";
}

