import re
import numpy as np


def extract_answer(response_tokens):
    """
    Extract the final numeric answer from response tokens.
    Priority: \\boxed{N} → #### N → last integer → None
    Returns int or None.
    """
    text = "".join(response_tokens)

    # priority 1: \\boxed{N} — LLaDA's natural format
    matches = re.findall(r'\\boxed\{(-?\d+)\}', text)
    if matches:
        return int(matches[-1])

    # priority 2: #### N — GSM8K standard
    matches = re.findall(r'####\s*(-?\d+)', text)
    if matches:
        return int(matches[-1])

    # priority 3: last standalone integer — fallback
    matches = re.findall(r'(?<!\d)(-?\d+)(?!\d)', text)
    if matches:
        return int(matches[-1])

    return None


def classify_failure_mode(response_str, extracted_answer, ground_truth):
    """
    Returns one of: 'CORRECT' | 'WRONG_ATTEMPT' | 'PARTIAL_REFUSAL' |
                    'FULL_REFUSAL' | 'DEGENERATE'
    """
    REFUSAL_PHRASES = [
        "i'm sorry", "i am sorry", "i apologize", "i cannot",
        "i can't", "i am unable to", "i'm unable to",
        "i am not able to", "i'm not able to", "as an ai",
        "as a language model",
    ]
    has_refusal = any(p in response_str.lower() for p in REFUSAL_PHRASES)

    if extracted_answer is not None and extracted_answer == ground_truth:
        return 'CORRECT'
    if has_refusal and extracted_answer is None:
        return 'FULL_REFUSAL'
    if has_refusal and extracted_answer is not None:
        return 'PARTIAL_REFUSAL'

    if extracted_answer is None:
        words = response_str.split()
        if len(response_str.strip()) < 20:
            return 'DEGENERATE'
        trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
        if trigrams and len(set(trigrams)) / len(trigrams) < 0.7:
            return 'DEGENERATE'
        return 'DEGENERATE'  # no answer, no refusal = degenerate

    return 'WRONG_ATTEMPT'


def find_connective_positions(response_tokens, connectives):
    """
    Return list of token indices where the token starts with a connective word.
    Handles BPE space prefixes (e.g. ' therefore') by stripping before matching.
    """
    return [
        i for i, tok in enumerate(response_tokens)
        if any(tok.strip().lower().startswith(c) for c in connectives)
    ]


def compute_connective_entropy(measurements, connective_pos):
    """
    Entropy_at_commit at connective positions.
    Returns dict with keys: 'entropies', 'mean', 'positions'.
    """
    if not connective_pos:
        return {'entropies': [], 'mean': None, 'positions': []}
    entropies = measurements['entropy_at_commit']
    vals = [float(entropies[i]) for i in connective_pos if i < len(entropies)]
    return {
        'entropies': vals,
        'mean':      float(np.mean(vals)) if vals else None,
        'positions': connective_pos,
    }


if __name__ == "__main__":
    test_cases = [
        (r"therefore \boxed{42}",              42,   "boxed primary"),
        (r"#### 42",                           42,   "hash primary"),
        (r"\boxed{10} ... \boxed{42}",         42,   "last boxed wins"),
        (r"#### 10 ... \boxed{42}",            42,   "boxed beats hash"),
        (r"the answer is 3 bolts",              3,   "fallback integer"),
        (r"I cannot answer this",            None,   "no number"),
        (r"\boxed{-5}",                        -5,   "negative"),
        (r"step 1: 10, step 2: 42\boxed{42}", 42,   "boxed beats stray integers"),
    ]

    all_pass = True
    for response, expected, description in test_cases:
        tokens = list(response)
        result = extract_answer(tokens)
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"{status}  [{description}]  got={result}  expected={expected}")

    print("\nAll tests passed." if all_pass else "\nSome tests FAILED.")
