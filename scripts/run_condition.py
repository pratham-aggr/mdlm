import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

"""
Usage:
    python scripts/run_condition.py --condition A
    python scripts/run_condition.py --condition B --n_problems 10  # for testing
"""
import argparse
import json
import time
import traceback

import numpy as np
import torch

from config import RESULTS_DIR, DATA_DIR, LOGICAL_CONNECTIVES
from model import load_model
from steering import extract_steering_vector, _get_layer
from generate import generate_with_measurements
from measure import (
    extract_answer,
    classify_failure_mode,
    find_connective_positions,
    compute_connective_entropy,
)
from conditions import CONDITIONS
import config

def run(condition_key, n_problems=None, args=None):
    assert condition_key in CONDITIONS, \
        f"Unknown condition {condition_key!r}. Choose from {list(CONDITIONS.keys())}"

    condition = CONDITIONS[condition_key]
    if args is not None and args.no_hidden_states:
        condition = dict(condition)        # copy — do not mutate CONDITIONS
        condition['save_hidden_states'] = False
        print("Hidden state saving disabled via --no_hidden_states flag.")
    out_dir   = os.path.join(RESULTS_DIR, f"condition_{condition_key}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Condition {condition_key}: {condition['description']}")
    print(f"Output dir: {out_dir}")

    # ---------------------------------------------------------------- load model
    model, tokenizer, MASK_ID = load_model()

    # --------------------------------------------------------- load steering vector
    vector_path = os.path.join(RESULTS_DIR, "steering_vector.pt")
    if condition['steer_layers']:
        assert os.path.exists(vector_path), \
            f"Steering vector not found at {vector_path}. Run extract_vector.py first."
        steering_vector = torch.load(vector_path).to(next(model.parameters()).device)
        print(f"Loaded steering vector, norm={steering_vector.norm():.4f}")
    else:
        steering_vector = None
        print("No steering vector needed for this condition.")

    # --------------------------------------------------------------- load problems
    problems_path = os.path.join(DATA_DIR, "gsm8k_200.jsonl")
    with open(problems_path) as f:
        problems = [json.loads(line) for line in f]
    if n_problems is not None:
        problems = problems[:n_problems]
    print(f"Loaded {len(problems)} problems.")

    # --------------------------------------------------------------- run loop
    results_path = os.path.join(out_dir, "results.jsonl")

    # load already-completed problem ids
    completed_ids = set()
    if os.path.exists(results_path):
        with open(results_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    completed_ids.add(r['problem_id'])
                except:
                    pass
        print(f"Resuming: {len(completed_ids)} problems already completed.")

    # load existing results
    results = []
    if os.path.exists(results_path):
        with open(results_path) as f:
            for line in f:
                try:
                    results.append(json.loads(line))
                except:
                    pass
    n_correct = sum(1 for r in results if r.get('failure_mode') == 'CORRECT')

    for i, problem in enumerate(problems):
        if problem['id'] in completed_ids:
            continue
        t0 = time.time()
        try:
            measurements, response_tokens = generate_with_measurements(
                model, tokenizer, MASK_ID,
                problem['question'],
                condition,
                steering_vector,
                config,
            )

            response_str     = "".join(response_tokens)
            extracted_answer = extract_answer(response_tokens)
            ground_truth     = problem['answer']
            failure_mode     = classify_failure_mode(
                                   response_str, extracted_answer, ground_truth)
            connective_pos   = find_connective_positions(
                                   response_tokens, LOGICAL_CONNECTIVES)
            connective_ent   = compute_connective_entropy(
                                   measurements, connective_pos)

            # extraction path logging
            if extracted_answer is None:
                extraction_path = 'none'
            elif r'\boxed' in response_str and str(extracted_answer) in response_str:
                extraction_path = 'boxed'
            elif '####' in response_str:
                extraction_path = 'hash'
            else:
                extraction_path = 'fallback'

            record = {
                'problem_id':          problem['id'],
                'ground_truth':        ground_truth,
                'extracted_answer':    extracted_answer,
                'failure_mode':        failure_mode,
                'extraction_path':     extraction_path,
                'response_length':     len(response_tokens),
                'mean_confidence':     float(measurements['confidence_at_unmask'].mean()),
                'mean_entropy_commit': float(measurements['entropy_at_commit'].mean()),
                'entropy_per_step':    measurements['entropy_per_step'].tolist(),
                'connective_entropy':  connective_ent,
                'n_connectives':       len(connective_pos),
            }

            if failure_mode == 'CORRECT':
                n_correct += 1

            # save hidden states separately if present
            if 'hidden_states' in measurements:
                hs_path = os.path.join(out_dir, f"hidden_states_{problem['id']:04d}.pt")
                torch.save(measurements['hidden_states'], hs_path)

        except Exception as e:
            print(f"  ERROR on problem {problem['id']}: {e}")
            traceback.print_exc()
            record = {
                'problem_id':       problem['id'],
                'ground_truth':     problem['answer'],
                'extracted_answer': None,
                'failure_mode':     'ERROR',
                'error':            str(e),
            }

        results.append(record)

        elapsed = time.time() - t0
        acc_so_far = n_correct / (i + 1)
        print(f"  [{i+1:>3}/{len(problems)}] "
              f"id={problem['id']:>4}  "
              f"gt={ground_truth:>4}  "
              f"pred={str(extracted_answer):>6}  "
              f"mode={record['failure_mode']:<16}  "
              f"path={extraction_path:<10}  "
              f"acc={acc_so_far:.3f}  "
              f"t={elapsed:.1f}s")

        # save results after every problem — crash-safe
        with open(results_path, 'a') as f:
            f.write(json.dumps(record) + '\n')

    # ----------------------------------------------------------------- summary
    print(f"\n{'='*60}")
    print(f"Condition {condition_key} complete.")
    print(f"Accuracy: {n_correct}/{len(problems)} = {n_correct/len(problems):.3f}")

    failure_counts = {}
    for r in results:
        fm = r['failure_mode']
        failure_counts[fm] = failure_counts.get(fm, 0) + 1
    print("Failure modes:")
    for fm, count in sorted(failure_counts.items()):
        print(f"  {fm:<20} {count:>4}  ({count/len(problems)*100:.1f}%)")

    extraction_counts = {}
    for r in results:
        ep = r.get('extraction_path', 'unknown')
        extraction_counts[ep] = extraction_counts.get(ep, 0) + 1
    print("Extraction paths:")
    for ep, count in sorted(extraction_counts.items()):
        print(f"  {ep:<20} {count:>4}  ({count/len(problems)*100:.1f}%)")

    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition',   required=True,
                        choices=list(CONDITIONS.keys()))
    parser.add_argument('--n_problems',  type=int, default=None,
                        help='Run on first N problems only. Omit for all 200.')
    parser.add_argument('--no_hidden_states', action='store_true',
        help='Override save_hidden_states to False regardless of condition config.')
    args = parser.parse_args()
    run(args.condition, args.n_problems, args)
