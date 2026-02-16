"""
Evaluation Script for Legal Q&A Model

Evaluates trained model on test set with:
- Citation accuracy
- ROUGE-L scores
- Language-specific performance
- Comprehensive evaluation report
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.metrics import LegalQAMetrics, LanguageSpecificMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluator for legal Q&A model"""

    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Initialize evaluator

        Args:
            model_path: Path to trained model
            device: Device to use (cuda/cpu)
        """
        logger.info(f"Loading model from {model_path}")

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
        self.model.eval()

        self.metrics = LegalQAMetrics()
        self.lang_metrics = LanguageSpecificMetrics()

        logger.info("Model loaded successfully")

    def generate_answer(self, question: str, max_new_tokens: int = 512,
                       temperature: float = 0.7) -> str:
        """
        Generate answer for a question

        Args:
            question: Input question
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated answer
        """
        # Format as chat
        messages = [
            {"role": "system", "content": "You are a specialized legal assistant for European Union law."},
            {"role": "user", "content": question}
        ]

        # Apply chat template
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(input_text, return_tensors='pt').to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract assistant response (after last "assistant" marker)
        if '<|start_header_id|>assistant<|end_header_id|>' in full_response:
            response = full_response.split('<|start_header_id|>assistant<|end_header_id|>')[-1]
        else:
            response = full_response[len(input_text):]

        return response.strip()

    def evaluate_dataset(self, dataset_path: str, max_samples: int = None) -> Dict:
        """
        Evaluate model on dataset

        Args:
            dataset_path: Path to evaluation dataset (JSONL)
            max_samples: Maximum number of samples to evaluate (None = all)

        Returns:
            Evaluation results dictionary
        """
        logger.info(f"Loading dataset from {dataset_path}")

        # Load dataset
        dataset = load_dataset('json', data_files=dataset_path, split='train')

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        logger.info(f"Evaluating on {len(dataset)} samples")

        all_metrics = []
        results = []

        for example in tqdm(dataset, desc="Evaluating"):
            question = example.get('question', '')
            reference_answer = example.get('answer', '')
            metadata = example.get('metadata', {})
            language = metadata.get('language', 'unknown')

            # Generate answer
            try:
                generated_answer = self.generate_answer(question)
            except Exception as e:
                logger.warning(f"Error generating answer: {e}")
                continue

            # Compute metrics
            metrics = self.metrics.compute_all_metrics(generated_answer, reference_answer)
            all_metrics.append(metrics)

            # Track language-specific
            self.lang_metrics.add_example(generated_answer, reference_answer, language)

            # Store result
            results.append({
                'question': question,
                'reference': reference_answer,
                'generated': generated_answer,
                'metrics': metrics,
                'metadata': metadata
            })

        # Aggregate metrics
        aggregated = self.metrics.aggregate_metrics(all_metrics)

        # Get language-specific summary
        lang_summary = self.lang_metrics.get_language_summary()

        # Compile final report
        report = {
            'overall_metrics': aggregated,
            'language_metrics': lang_summary,
            'num_samples': len(results),
            'sample_results': results[:10]  # Include first 10 for inspection
        }

        return report

    def save_report(self, report: Dict, output_path: str):
        """Save evaluation report to JSON"""
        logger.info(f"Saving report to {output_path}")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info("Report saved successfully")

        # Print summary
        self.print_summary(report)

    def print_summary(self, report: Dict):
        """Print evaluation summary"""
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)

        print(f"\nSamples evaluated: {report['num_samples']}")

        print("\nOverall Metrics:")
        overall = report['overall_metrics']
        for metric, value in sorted(overall.items()):
            if 'mean' in metric:
                print(f"  {metric}: {value:.4f}")

        print("\nLanguage-Specific Performance:")
        lang_metrics = report['language_metrics']
        for lang, metrics in sorted(lang_metrics.items()):
            print(f"\n  {lang.upper()}:")
            for metric, value in sorted(metrics.items()):
                if 'mean' in metric:
                    print(f"    {metric}: {value:.4f}")

        print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Legal Q&A Model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--eval_dataset', type=str, required=True,
                       help='Path to evaluation dataset (JSONL)')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output path for evaluation report (JSON)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = ModelEvaluator(args.model_path, device=args.device)

    # Run evaluation
    report = evaluator.evaluate_dataset(args.eval_dataset, max_samples=args.max_samples)

    # Save report
    evaluator.save_report(report, args.output_file)

    logger.info("Evaluation complete!")


if __name__ == '__main__':
    main()
