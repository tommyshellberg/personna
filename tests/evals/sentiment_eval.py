#!/usr/bin/env python3
"""
Sentiment Analysis Evaluation Runner

Evaluates the SentimentAnalyzer against a curated test dataset using Langsmith.

Usage:
    # Run evaluation (requires LANGCHAIN_API_KEY)
    python tests/evals/sentiment_eval.py

    # Run without Langsmith (local-only mode)
    python tests/evals/sentiment_eval.py --local

Environment variables:
    LANGCHAIN_API_KEY: Langsmith API key
    LANGCHAIN_TRACING_V2: Set to 'true' to enable tracing
    LANGCHAIN_PROJECT: Project name (default: 'reddit-sentiment')
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.sentiment_analyzer import SentimentAnalyzer


def load_test_dataset(path: Path) -> List[Dict[str, Any]]:
    """Load test cases from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data['cases']


def score_in_range(predicted_score: float, expected_min: float, expected_max: float) -> bool:
    """Check if predicted score falls within expected range."""
    return expected_min <= predicted_score <= expected_max


def classification_correct(predicted_score: float, expected_sentiment: str) -> bool:
    """Check if sentiment classification matches expected category."""
    if expected_sentiment == "positive":
        return predicted_score > 0.2
    elif expected_sentiment == "negative":
        return predicted_score < -0.1
    else:  # neutral
        return -0.3 <= predicted_score <= 0.4


def run_local_evaluation(test_cases: List[Dict], analyzer: SentimentAnalyzer) -> Dict[str, Any]:
    """Run evaluation locally without Langsmith."""
    results = {
        'total': len(test_cases),
        'score_in_range': 0,
        'classification_correct': 0,
        'details': []
    }

    print(f"\nRunning evaluation on {len(test_cases)} test cases...\n")
    print("-" * 80)

    for case in test_cases:
        # Run sentiment analysis
        output = analyzer.analyze_single(
            comment=case['comment'],
            post_title=case['post_title'],
            post_body=case.get('post_body', '')
        )

        predicted_score = output['score']
        rationale = output['rationale']

        # Evaluate
        in_range = score_in_range(
            predicted_score,
            case['expected_score_min'],
            case['expected_score_max']
        )
        correct_class = classification_correct(
            predicted_score,
            case['expected_sentiment']
        )

        if in_range:
            results['score_in_range'] += 1
        if correct_class:
            results['classification_correct'] += 1

        # Status emoji
        status = "✓" if (in_range and correct_class) else "✗"

        # Store details
        detail = {
            'id': case['id'],
            'expected': case['expected_sentiment'],
            'expected_range': f"[{case['expected_score_min']}, {case['expected_score_max']}]",
            'predicted_score': predicted_score,
            'rationale': rationale,
            'in_range': in_range,
            'correct_class': correct_class
        }
        results['details'].append(detail)

        # Print result
        range_str = f"[{case['expected_score_min']:+.1f}, {case['expected_score_max']:+.1f}]"
        print(f"{status} {case['id']:<25} | expected: {case['expected_sentiment']:<8} {range_str} | "
              f"got: {predicted_score:+.2f} | {rationale[:30]}")

    # Summary
    print("-" * 80)
    print(f"\n📊 Results Summary:")
    print(f"   Score in range:        {results['score_in_range']}/{results['total']} "
          f"({100*results['score_in_range']/results['total']:.1f}%)")
    print(f"   Classification correct: {results['classification_correct']}/{results['total']} "
          f"({100*results['classification_correct']/results['total']:.1f}%)")

    return results


def run_langsmith_evaluation(test_cases: List[Dict], analyzer: SentimentAnalyzer) -> Dict[str, Any]:
    """Run evaluation using Langsmith for tracking."""
    try:
        from langsmith import Client
    except ImportError:
        print("Langsmith not installed. Run: pip install langsmith")
        sys.exit(1)

    # Check for API key
    if not os.environ.get('LANGCHAIN_API_KEY'):
        print("⚠️  LANGCHAIN_API_KEY not set. Running in local mode instead.")
        return run_local_evaluation(test_cases, analyzer)

    client = Client()
    dataset_name = "reddit-sentiment-eval"

    # Create or get dataset
    try:
        datasets = list(client.list_datasets(dataset_name=dataset_name))
        if datasets:
            dataset = datasets[0]
            print(f"Using existing dataset: {dataset_name}")
        else:
            dataset = client.create_dataset(dataset_name=dataset_name)
            print(f"Created new dataset: {dataset_name}")

            # Add examples
            examples = [
                {
                    "inputs": {
                        "comment": case['comment'],
                        "post_title": case['post_title'],
                        "post_body": case.get('post_body', '')
                    },
                    "outputs": {
                        "expected_sentiment": case['expected_sentiment'],
                        "expected_score_min": case['expected_score_min'],
                        "expected_score_max": case['expected_score_max']
                    },
                    "metadata": {"id": case['id']}
                }
                for case in test_cases
            ]

            client.create_examples(
                dataset_id=dataset.id,
                inputs=[e["inputs"] for e in examples],
                outputs=[e["outputs"] for e in examples],
                metadata=[e["metadata"] for e in examples]
            )
            print(f"Added {len(examples)} examples to dataset")

    except Exception as e:
        print(f"Error with Langsmith dataset: {e}")
        print("Falling back to local evaluation...")
        return run_local_evaluation(test_cases, analyzer)

    # Define the target function for evaluation
    def target(inputs: dict) -> dict:
        return analyzer.analyze_single(
            comment=inputs['comment'],
            post_title=inputs['post_title'],
            post_body=inputs.get('post_body', '')
        )

    # Define evaluators
    def score_range_evaluator(run, example) -> dict:
        """Check if score is within expected range."""
        predicted = run.outputs.get('score', 0)
        expected_min = example.outputs.get('expected_score_min', -1)
        expected_max = example.outputs.get('expected_score_max', 1)
        passed = expected_min <= predicted <= expected_max
        return {
            "key": "score_in_range",
            "score": 1.0 if passed else 0.0,
            "comment": f"Predicted {predicted:.2f}, expected [{expected_min}, {expected_max}]"
        }

    def classification_evaluator(run, example) -> dict:
        """Check if classification matches expected sentiment."""
        predicted = run.outputs.get('score', 0)
        expected = example.outputs.get('expected_sentiment', 'neutral')

        if expected == "positive":
            passed = predicted > 0.2
        elif expected == "negative":
            passed = predicted < -0.1
        else:
            passed = -0.3 <= predicted <= 0.4

        return {
            "key": "classification_correct",
            "score": 1.0 if passed else 0.0,
            "comment": f"Predicted {predicted:.2f}, expected {expected}"
        }

    # Run evaluation
    print(f"\n🔬 Running Langsmith evaluation on dataset: {dataset_name}")
    print("   View results at: https://smith.langchain.com\n")

    try:
        results = client.evaluate(
            target,
            data=dataset_name,
            evaluators=[score_range_evaluator, classification_evaluator],
            experiment_prefix="sentiment-eval",
            max_concurrency=2,
        )

        print("\n✅ Evaluation complete! Check Langsmith for detailed results.")
        return {"langsmith_experiment": str(results)}

    except Exception as e:
        print(f"Langsmith evaluation failed: {e}")
        print("Falling back to local evaluation...")
        return run_local_evaluation(test_cases, analyzer)


def main():
    parser = argparse.ArgumentParser(description="Run sentiment analysis evaluation")
    parser.add_argument('--local', action='store_true',
                        help='Run locally without Langsmith')
    parser.add_argument('--config', type=Path,
                        default=project_root / 'config' / 'settings.yaml',
                        help='Path to config file')
    parser.add_argument('--dataset', type=Path,
                        default=Path(__file__).parent / 'datasets' / 'sentiment_cases.json',
                        help='Path to test dataset')
    args = parser.parse_args()

    # Load config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Force temperature=0 for evaluation reproducibility
    config['ollama']['temperature'] = 0

    # Initialize analyzer
    analyzer = SentimentAnalyzer(config)

    # Load test cases
    test_cases = load_test_dataset(args.dataset)
    print(f"📂 Loaded {len(test_cases)} test cases from {args.dataset}")

    # Run evaluation
    if args.local:
        results = run_local_evaluation(test_cases, analyzer)
    else:
        results = run_langsmith_evaluation(test_cases, analyzer)

    # Return success/failure based on accuracy
    if 'classification_correct' in results:
        accuracy = results['classification_correct'] / results['total']
        sys.exit(0 if accuracy >= 0.8 else 1)


if __name__ == '__main__':
    main()
