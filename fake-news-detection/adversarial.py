"""
Adversarial Testing Module
Tests model robustness against paraphrased and adversarial inputs.
"""
import random
import numpy as np


# Synonym replacements for adversarial paraphrasing
SYNONYM_MAP = {
    'claims': ['asserts', 'states', 'says', 'declares', 'maintains'],
    'scientists': ['researchers', 'experts', 'scholars', 'academics'],
    'government': ['state', 'administration', 'authorities', 'officials'],
    'study': ['research', 'investigation', 'analysis', 'examination'],
    'breakthrough': ['discovery', 'advancement', 'milestone', 'achievement'],
    'miracle': ['wonder', 'marvel', 'phenomenon', 'spectacle'],
    'conspiracy': ['scheme', 'plot', 'collusion', 'intrigue'],
    'dangerous': ['hazardous', 'risky', 'perilous', 'unsafe'],
    'proves': ['demonstrates', 'confirms', 'validates', 'establishes'],
    'secret': ['hidden', 'concealed', 'covert', 'clandestine'],
    'fake': ['bogus', 'fraudulent', 'counterfeit', 'phony'],
    'real': ['genuine', 'authentic', 'legitimate', 'actual'],
    'cure': ['remedy', 'treatment', 'therapy', 'solution'],
    'cause': ['lead to', 'result in', 'produce', 'trigger'],
    'discover': ['find', 'uncover', 'reveal', 'identify'],
    'claim': ['assert', 'state', 'declare', 'maintain'],
    'widespread': ['extensive', 'pervasive', 'prevalent', 'ubiquitous'],
    'debunked': ['disproved', 'refuted', 'dismissed', 'rejected'],
    'conspiracy theorists': ['skeptics', 'doubters', 'critics', 'questioners'],
    'viral': ['trending', 'popular', 'widely shared', 'circulating'],
    'shocking': ['startling', 'astonishing', 'surprising', 'alarming'],
    'dangerous': ['hazardous', 'harmful', 'risky', 'threatening'],
    'proof': ['evidence', 'confirmation', 'demonstration', 'verification'],
    'exposed': ['revealed', 'uncovered', 'disclosed', 'brought to light'],
    'pharmaceutical': ['drug', 'medication', 'medical', 'biotech'],
    'researchers found': ['studies show', 'data indicates', 'analysis reveals'],
    'according to': ['as reported by', 'based on', 'per'],
    'officials said': ['authorities stated', 'spokesperson confirmed', 'sources reported'],
}


def paraphrase_text(text, num_swaps=3):
    """
    Paraphrase text by swapping words with synonyms.

    Args:
        text: Input text.
        num_swaps: Number of word replacements.

    Returns:
        Paraphrased text.
    """
    words = text.split()
    swaps_done = 0
    indices_to_swap = list(range(len(words)))
    random.shuffle(indices_to_swap)

    for idx in indices_to_swap:
        if swaps_done >= num_swaps:
            break
        word_lower = words[idx].lower().rstrip('.,!?;:')
        if word_lower in SYNONYM_MAP:
            synonyms = SYNONYM_MAP[word_lower]
            new_word = random.choice(synonyms)
            # Preserve capitalization
            if words[idx][0].isupper():
                new_word = new_word.capitalize()
            # Preserve trailing punctuation
            if words[idx][-1] in '.,!?;:':
                new_word += words[idx][-1]
            words[idx] = new_word
            swaps_done += 1

    return ' '.join(words)


def add_noise(text, noise_level=0.1):
    """
    Add random noise to text (random word insertions/deletions).

    Args:
        text: Input text.
        noise_level: Fraction of words to modify.

    Returns:
        Noisy text.
    """
    words = text.split()
    n_modify = max(1, int(len(words) * noise_level))

    # Randomly remove some words
    for _ in range(n_modify // 2):
        if len(words) > 5:
            idx = random.randint(0, len(words) - 1)
            words.pop(idx)

    # Randomly insert common words
    filler_words = ['the', 'a', 'is', 'was', 'are', 'has', 'have', 'been', 'will']
    for _ in range(n_modify // 2):
        idx = random.randint(0, len(words))
        words.insert(idx, random.choice(filler_words))

    return ' '.join(words)


def generate_adversarial_examples(text, label, n_variants=5):
    """
    Generate adversarial variants of a text.

    Args:
        text: Original text.
        label: Original label (0=real, 1=fake).
        n_variants: Number of variants to generate.

    Returns:
        List of (text, label, variant_type) tuples.
    """
    variants = []

    # Paraphrase variants
    for i in range(n_variants // 2):
        paraphrased = paraphrase_text(text, num_swaps=random.randint(2, 5))
        variants.append((paraphrased, label, f'paraphrase_{i+1}'))

    # Noise variants
    for i in range(n_variants // 2):
        noisy = add_noise(text, noise_level=random.uniform(0.05, 0.15))
        variants.append((noisy, label, f'noise_{i+1}'))

    # Combined paraphrase + noise
    combined = paraphrase_text(text, num_swaps=3)
    combined = add_noise(combined, noise_level=0.1)
    variants.append((combined, label, 'combined'))

    return variants


def adversarial_test(model, vectorizer, preprocessor, test_cases, n_variants=3):
    """
    Run adversarial robustness test.

    Args:
        model: Trained classifier.
        vectorizer: Fitted TF-IDF vectorizer.
        preprocessor: Text preprocessor.
        test_cases: List of (text, label) tuples.
        n_variants: Number of adversarial variants per example.

    Returns:
        Dict with original and adversarial accuracy results.
    """
    results = {
        'original_correct': 0,
        'adversarial_correct': 0,
        'total_original': len(test_cases),
        'total_adversarial': 0,
        'failures': []
    }

    for text, true_label in test_cases:
        # Test original
        cleaned = preprocessor.clean_text(text)
        features = vectorizer.transform([cleaned])
        pred = model.predict(features)[0]

        if pred == true_label:
            results['original_correct'] += 1

        # Test adversarial variants
        variants = generate_adversarial_examples(text, true_label, n_variants)
        for variant_text, variant_label, variant_type in variants:
            results['total_adversarial'] += 1
            cleaned_var = preprocessor.clean_text(variant_text)
            features_var = vectorizer.transform([cleaned_var])
            pred_var = model.predict(features_var)[0]

            if pred_var == variant_label:
                results['adversarial_correct'] += 1
            else:
                results['failures'].append({
                    'original': text[:80],
                    'variant': variant_text[:80],
                    'variant_type': variant_type,
                    'true_label': 'REAL' if variant_label == 0 else 'FAKE',
                    'predicted': 'REAL' if pred_var == 0 else 'FAKE'
                })

    results['original_accuracy'] = results['original_correct'] / results['total_original']
    results['adversarial_accuracy'] = (
        results['adversarial_correct'] / max(1, results['total_adversarial'])
    )
    results['robustness_gap'] = results['original_accuracy'] - results['adversarial_accuracy']

    return results
