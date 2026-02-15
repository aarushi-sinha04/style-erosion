"""
Synonym Replacement Attack for Authorship Verification
========================================================
Word-level adversarial attack using WordNet synonyms.
Replaces content words (nouns, verbs, adjectives, adverbs) with
their WordNet synonyms at a configurable rate.

Unlike T5 paraphrasing (sentence-level rewriting), synonym replacement
operates at the word level, preserving syntactic structure but altering
lexical choices — testing a fundamentally different attack surface.

Usage:
    # Single text
    from attacks.synonym_attack import SynonymAttacker
    attacker = SynonymAttacker(replacement_rate=0.15)
    attacked = attacker.attack("The quick brown fox jumps over the lazy dog.")

    # Batch generation
    python -m attacks.synonym_attack --num_samples 200
"""
import sys
import os
import re
import json
import random
import argparse
import numpy as np
from tqdm import tqdm

import nltk
from nltk.corpus import wordnet as wn

# Ensure WordNet data is available
try:
    wn.synsets('test')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

try:
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
except Exception:
    nlp = None
    print("Warning: spacy en_core_web_sm not available. Using NLTK POS tags.")
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    from nltk import pos_tag, word_tokenize


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ==============================================================================
# POS Mapping
# ==============================================================================
# Map spaCy/Penn Treebank POS to WordNet POS
POS_MAP = {
    'NOUN': wn.NOUN, 'NN': wn.NOUN, 'NNS': wn.NOUN,
    'NNP': wn.NOUN, 'NNPS': wn.NOUN,
    'VERB': wn.VERB, 'VB': wn.VERB, 'VBD': wn.VERB,
    'VBG': wn.VERB, 'VBN': wn.VERB, 'VBP': wn.VERB, 'VBZ': wn.VERB,
    'ADJ': wn.ADJ, 'JJ': wn.ADJ, 'JJR': wn.ADJ, 'JJS': wn.ADJ,
    'ADV': wn.ADV, 'RB': wn.ADV, 'RBR': wn.ADV, 'RBS': wn.ADV,
}

# Common function words to skip (attacking these doesn't change style meaningfully)
SKIP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'must', 'need', 'dare',
    'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
    'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'between', 'under', 'since', 'without', 'about', 'against',
    'and', 'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
    'neither', 'each', 'every', 'all', 'any', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'only', 'own', 'same', 'than',
    'too', 'very', 's', 't', 'just', 'don', 'now', 'i', 'he', 'she',
    'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
    'his', 'its', 'our', 'their', 'this', 'that', 'these', 'those',
    'what', 'which', 'who', 'whom', 'when', 'where', 'why', 'how',
}


class SynonymAttacker:
    """
    Word-level adversarial attack using WordNet synonym replacement.

    Replaces content words with synonyms while preserving:
    - Syntactic structure (POS preserved)
    - Semantic meaning (synonyms from WordNet)
    - Readability (single-word replacements only)

    Args:
        replacement_rate: fraction of eligible words to replace (default 0.15)
        seed: random seed for reproducibility
    """

    def __init__(self, replacement_rate: float = 0.15, seed: int = 42):
        self.replacement_rate = replacement_rate
        self.rng = random.Random(seed)
        self._synonym_cache = {}

    def _get_synonyms(self, word: str, pos: str = None) -> list:
        """Get WordNet synonyms for a word, optionally filtered by POS."""
        cache_key = (word.lower(), pos)
        if cache_key in self._synonym_cache:
            return self._synonym_cache[cache_key]

        synonyms = set()
        wn_pos = POS_MAP.get(pos)

        synsets = wn.synsets(word.lower(), pos=wn_pos) if wn_pos else wn.synsets(word.lower())

        for syn in synsets:
            for lemma in syn.lemmas():
                name = lemma.name().replace('_', ' ')
                # Only single-word synonyms, different from original
                if ' ' not in name and name.lower() != word.lower():
                    synonyms.add(name)

        result = list(synonyms)
        self._synonym_cache[cache_key] = result
        return result

    def _tokenize_with_pos(self, text: str):
        """
        Tokenize text and get POS tags.
        Returns list of (token_text, pos_tag, is_replaceable).
        """
        if nlp is not None:
            doc = nlp(text)
            tokens = []
            for tok in doc:
                pos = tok.pos_
                is_content = (pos in POS_MAP and
                              tok.text.lower() not in SKIP_WORDS and
                              tok.is_alpha and
                              len(tok.text) > 2)
                tokens.append((tok.text, pos, is_content, tok.whitespace_))
            return tokens
        else:
            # Fallback: NLTK
            words = word_tokenize(text)
            tagged = pos_tag(words)
            tokens = []
            for word, tag in tagged:
                is_content = (tag in POS_MAP and
                              word.lower() not in SKIP_WORDS and
                              word.isalpha() and
                              len(word) > 2)
                tokens.append((word, tag, is_content, ' '))
            return tokens

    def attack(self, text: str) -> str:
        """
        Apply synonym replacement attack to a text.

        Returns the modified text with some content words replaced by synonyms.
        """
        if not text or not text.strip():
            return text

        tokens = self._tokenize_with_pos(text)

        # Find replaceable positions
        replaceable_indices = [i for i, (_, _, is_repl, _) in enumerate(tokens) if is_repl]

        if not replaceable_indices:
            return text

        # Determine how many words to replace
        n_replace = max(1, int(len(replaceable_indices) * self.replacement_rate))
        n_replace = min(n_replace, len(replaceable_indices))

        # Randomly select positions to replace
        replace_indices = set(self.rng.sample(replaceable_indices, n_replace))

        # Build result
        result_parts = []
        n_replaced = 0

        for i, (word, pos, _, ws) in enumerate(tokens):
            if i in replace_indices:
                synonyms = self._get_synonyms(word, pos)
                if synonyms:
                    replacement = self.rng.choice(synonyms)
                    # Preserve capitalization
                    if word[0].isupper():
                        replacement = replacement.capitalize()
                    elif word.isupper():
                        replacement = replacement.upper()
                    result_parts.append(replacement)
                    n_replaced += 1
                else:
                    result_parts.append(word)
            else:
                result_parts.append(word)

            # Add whitespace
            if nlp is not None:
                result_parts.append(ws)
            else:
                result_parts.append(' ')

        result = ''.join(result_parts).strip()
        return result

    def attack_batch(self, texts: list) -> list:
        """Attack a batch of texts."""
        return [self.attack(t) for t in texts]

    def get_stats(self, original: str, attacked: str) -> dict:
        """Compute statistics about the attack."""
        orig_words = original.lower().split()
        atk_words = attacked.lower().split()

        if not orig_words:
            return {'n_original': 0, 'n_changed': 0, 'change_rate': 0.0}

        # Word-level changes
        min_len = min(len(orig_words), len(atk_words))
        n_changed = sum(1 for i in range(min_len) if orig_words[i] != atk_words[i])
        n_changed += abs(len(orig_words) - len(atk_words))

        return {
            'n_original_words': len(orig_words),
            'n_attacked_words': len(atk_words),
            'n_changed': n_changed,
            'change_rate': round(n_changed / len(orig_words), 4),
        }


# ==============================================================================
# Batch Generation
# ==============================================================================
def generate_synonym_adversarial_cache(num_samples=200, seed=42):
    """
    Generate synonym-attacked adversarial samples from PAN22 positive pairs.
    Mirrors the format of data/eval_adversarial_cache.jsonl.
    """
    from utils.data_loader_scie import PAN22Loader

    output_file = "data/synonym_adversarial_cache.jsonl"
    os.makedirs("data", exist_ok=True)

    print(f"Generating {num_samples} synonym-attacked samples...")

    # Load PAN22 positive pairs
    loader = PAN22Loader(
        "pan22-authorship-verification-training.jsonl",
        "pan22-authorship-verification-training-truth.jsonl"
    )
    loader.load(limit=num_samples * 4)
    t1, t2, labels = loader.create_pairs(num_pairs=num_samples * 2)

    # Filter to positive (same-author) pairs only
    positive_indices = [i for i, l in enumerate(labels) if l == 1]
    positive_indices = positive_indices[:num_samples]

    if len(positive_indices) < num_samples:
        print(f"  Warning: only {len(positive_indices)} positive pairs available")

    attacker = SynonymAttacker(replacement_rate=0.15, seed=seed)

    results = []
    for idx in tqdm(positive_indices, desc="Attacking"):
        anchor = t1[idx]
        positive = t2[idx]
        attacked = attacker.attack(positive)

        stats = attacker.get_stats(positive, attacked)

        entry = {
            'anchor': anchor,
            'positive': positive,
            'attacked': attacked,
            'attack_type': 'synonym_replacement',
            'replacement_rate': 0.15,
            'stats': stats,
        }
        results.append(entry)

    # Save
    with open(output_file, 'w') as f:
        for entry in results:
            f.write(json.dumps(entry) + '\n')

    print(f"  Saved {len(results)} samples to {output_file}")

    # Print stats summary
    avg_change_rate = np.mean([r['stats']['change_rate'] for r in results])
    print(f"  Average word change rate: {avg_change_rate:.2%}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Synonym Replacement Attack")
    parser.add_argument("--num_samples", type=int, default=200,
                        help="Number of adversarial samples to generate")
    parser.add_argument("--rate", type=float, default=0.15,
                        help="Replacement rate (fraction of content words)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--demo", action="store_true",
                        help="Run a quick demo on example text")
    args = parser.parse_args()

    if args.demo:
        print("=" * 60)
        print("SYNONYM ATTACK DEMO")
        print("=" * 60)
        attacker = SynonymAttacker(replacement_rate=args.rate, seed=args.seed)

        examples = [
            "The quick brown fox jumps over the lazy dog.",
            "She carefully examined the intricate pattern on the ancient vase.",
            "The researchers discovered a significant correlation between the variables.",
        ]

        for text in examples:
            attacked = attacker.attack(text)
            stats = attacker.get_stats(text, attacked)
            print(f"\nOriginal:  {text}")
            print(f"Attacked:  {attacked}")
            print(f"Stats:     {stats}")

    else:
        generate_synonym_adversarial_cache(
            num_samples=args.num_samples, seed=args.seed)


if __name__ == "__main__":
    main()
