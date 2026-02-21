"""
Model Comparison Script for EUR-Lex Q&A

Compares base LLaMA 3.3 70B vs fine-tuned model on test set:
- Sequential model loading for memory safety
- Batch inference on test questions
- Comprehensive metrics computation
- Side-by-side comparison reports

Output: JSON results, Markdown report, CSV predictions
"""

import sys
import json
import gc
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.metrics import LegalQAMetrics, LanguageSpecificMetrics
from src.evaluation.report_generator import generate_markdown_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelComparator:
    """Compare base and fine-tuned models on EUR-Lex Q&A"""

    def __init__(self, base_model_path: str, finetuned_model_path: str, device: str = 'cuda'):
        """
        Initialize comparator

        Args:
            base_model_path: Path to base model (e.g., meta-llama/Llama-3.3-70B-Instruct)
            finetuned_model_path: Path to fine-tuned model (e.g., ./checkpoints/sft/final)
            device: Device to use (cuda/cpu)
        """
        self.base_model_path = base_model_path
        self.finetuned_model_path = finetuned_model_path
        self.device = device
        self.metrics = LegalQAMetrics()

    def load_model(self, model_path: str):
        """Load model with FP4 quantization"""
        logger.info(f"Loading model from {model_path}...")

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Check GPU memory before loading
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            logger.info(f"GPU memory allocated before loading: {memory_allocated:.2f} GB")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            load_in_4bit=True
        )
        model.eval()

        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            logger.info(f"GPU memory allocated after loading: {memory_allocated:.2f} GB")

        logger.info("Model loaded successfully")
        return model, tokenizer

    def clear_model_memory(self, model, tokenizer):
        """Aggressively clear model from memory"""
        logger.info("Clearing model from memory...")

        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            logger.info(f"GPU memory after cleanup: {memory_allocated:.2f} GB")

    def generate_answer(self, model, tokenizer, question: str,
                       max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate answer for a question"""
        # Format as chat
        messages = [
            {"role": "system", "content": "You are a specialized legal assistant for European Union law."},
            {"role": "user", "content": question}
        ]

        # Apply chat template
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = tokenizer(input_text, return_tensors='pt').to(self.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract assistant response
        if '<|start_header_id|>assistant<|end_header_id|>' in full_response:
            response = full_response.split('<|start_header_id|>assistant<|end_header_id|>')[-1]
        else:
            response = full_response[len(input_text):]

        return response.strip()

    def run_inference(self, model, tokenizer, dataset, batch_size: int = 8,
                     max_new_tokens: int = 512, temperature: float = 0.7) -> List[Dict]:
        """Run inference on dataset"""
        results = []
        failures = 0

        logger.info(f"Running inference on {len(dataset)} samples...")

        for i, example in enumerate(tqdm(dataset, desc="Generating answers")):
            question = example.get('question', '')
            reference_answer = example.get('answer', '')
            language = example.get('language', 'unknown')
            metadata = example.get('metadata', {})

            try:
                generated_answer = self.generate_answer(
                    model, tokenizer, question, max_new_tokens, temperature
                )
            except Exception as e:
                logger.warning(f"Error generating answer for sample {i}: {e}")
                generated_answer = ""
                failures += 1

            results.append({
                'sample_id': i,
                'question': question,
                'reference': reference_answer,
                'generated': generated_answer,
                'language': language,
                'metadata': metadata
            })

        logger.info(f"Inference complete. Failures: {failures}/{len(dataset)}")
        return results

    def compute_metrics(self, results: List[Dict]) -> Dict:
        """Compute metrics for all results"""
        logger.info("Computing metrics...")

        # Initialize language-specific metrics
        lang_metrics = LanguageSpecificMetrics()

        # Compute metrics for each sample
        for result in results:
            generated = result['generated']
            reference = result['reference']
            language = result['language']

            # Compute all metrics
            metrics = self.metrics.compute_all_metrics(generated, reference)
            result['metrics'] = metrics

            # Track language-specific
            lang_metrics.add_example(generated, reference, language)

        # Aggregate overall metrics
        all_metrics = [r['metrics'] for r in results]
        overall_metrics = self.metrics.aggregate_metrics(all_metrics)

        # Get language-specific summary
        lang_summary = lang_metrics.get_language_summary()

        return {
            'overall': overall_metrics,
            'by_language': lang_summary
        }

    def compare_results(self, base_results: List[Dict], ft_results: List[Dict]) -> Dict:
        """Compare base and fine-tuned model results"""
        logger.info("Computing comparison metrics...")

        comparison = {
            'num_samples': len(base_results),
            'base_metrics': self.compute_metrics(base_results),
            'finetuned_metrics': self.compute_metrics(ft_results),
            'sample_comparisons': []
        }

        # Compute deltas
        base_overall = comparison['base_metrics']['overall']
        ft_overall = comparison['finetuned_metrics']['overall']

        deltas = {}
        relative_improvements = {}

        for metric in ['citation_accuracy_mean', 'article_accuracy_mean', 'rouge_l_mean', 'exact_match_mean']:
            if metric in base_overall and metric in ft_overall:
                delta = ft_overall[metric] - base_overall[metric]
                deltas[metric] = delta

                # Relative improvement (avoid division by zero)
                if base_overall[metric] > 0:
                    relative_improvements[metric] = (delta / base_overall[metric]) * 100
                else:
                    relative_improvements[metric] = 0.0

        comparison['deltas'] = deltas
        comparison['relative_improvements'] = relative_improvements

        # Win/loss/tie analysis
        wins = 0
        losses = 0
        ties = 0

        for base_res, ft_res in zip(base_results, ft_results):
            base_score = base_res['metrics'].get('rouge_l', 0)
            ft_score = ft_res['metrics'].get('rouge_l', 0)

            if ft_score > base_score + 0.05:  # Threshold for meaningful difference
                winner = 'finetuned'
                wins += 1
            elif base_score > ft_score + 0.05:
                winner = 'base'
                losses += 1
            else:
                winner = 'tie'
                ties += 1

            comparison['sample_comparisons'].append({
                'sample_id': base_res['sample_id'],
                'question': base_res['question'],
                'language': base_res['language'],
                'base_prediction': base_res['generated'],
                'finetuned_prediction': ft_res['generated'],
                'base_metrics': base_res['metrics'],
                'finetuned_metrics': ft_res['metrics'],
                'winner': winner
            })

        comparison['win_loss_tie'] = {
            'finetuned_wins': wins,
            'base_wins': losses,
            'ties': ties,
            'finetuned_win_rate': wins / len(base_results) if base_results else 0
        }

        return comparison

    def run_comparison(self, test_dataset_path: str, output_dir: str,
                      batch_size: int = 8, max_new_tokens: int = 512,
                      temperature: float = 0.7):
        """Run full comparison workflow"""
        logger.info("=" * 80)
        logger.info("EUR-Lex Model Comparison")
        logger.info("=" * 80)

        # Load test dataset
        logger.info(f"\n[1/7] Loading test dataset from {test_dataset_path}")
        dataset = load_dataset('json', data_files=test_dataset_path, split='train')
        logger.info(f"Loaded {len(dataset)} test samples")

        # Phase 1: Base model
        logger.info("\n[2/7] Loading base model")
        base_model, base_tokenizer = self.load_model(self.base_model_path)

        logger.info("\n[3/7] Running base model inference")
        base_results = self.run_inference(
            base_model, base_tokenizer, dataset, batch_size, max_new_tokens, temperature
        )

        logger.info("\n[4/7] Clearing base model from memory")
        self.clear_model_memory(base_model, base_tokenizer)

        # Phase 2: Fine-tuned model
        logger.info("\n[5/7] Loading fine-tuned model")
        ft_model, ft_tokenizer = self.load_model(self.finetuned_model_path)

        logger.info("\n[6/7] Running fine-tuned model inference")
        ft_results = self.run_inference(
            ft_model, ft_tokenizer, dataset, batch_size, max_new_tokens, temperature
        )

        logger.info("\n[7/7] Computing comparison metrics")
        comparison = self.compare_results(base_results, ft_results)

        # Clear fine-tuned model
        self.clear_model_memory(ft_model, ft_tokenizer)

        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_path = output_path / "comparison_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        logger.info(f"\nSaved JSON results to: {json_path}")

        # Save CSV
        csv_path = output_path / "predictions.csv"
        self.save_predictions_csv(comparison['sample_comparisons'], csv_path)
        logger.info(f"Saved predictions CSV to: {csv_path}")

        # Generate Markdown report
        md_path = output_path / "comparison_report.md"
        generate_markdown_report(comparison, md_path)
        logger.info(f"Saved Markdown report to: {md_path}")

        # Print summary
        self.print_summary(comparison)

        return comparison

    def save_predictions_csv(self, sample_comparisons: List[Dict], output_path: Path):
        """Save predictions to CSV"""
        import csv

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'sample_id', 'language', 'question', 'reference',
                'base_prediction', 'ft_prediction',
                'base_citation_acc', 'ft_citation_acc',
                'base_rouge_l', 'ft_rouge_l',
                'winner'
            ])

            # Data
            for sample in sample_comparisons:
                writer.writerow([
                    sample['sample_id'],
                    sample['language'],
                    sample['question'][:100],  # Truncate for readability
                    sample.get('base_prediction', '')[:100],
                    sample.get('finetuned_prediction', '')[:100],
                    sample['base_metrics'].get('citation_accuracy', 0),
                    sample['finetuned_metrics'].get('citation_accuracy', 0),
                    sample['base_metrics'].get('rouge_l', 0),
                    sample['finetuned_metrics'].get('rouge_l', 0),
                    sample['winner']
                ])

    def print_summary(self, comparison: Dict):
        """Print comparison summary to console"""
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)

        print(f"\nSamples evaluated: {comparison['num_samples']}")

        # Overall metrics
        print("\n" + "-" * 80)
        print("OVERALL PERFORMANCE")
        print("-" * 80)

        base = comparison['base_metrics']['overall']
        ft = comparison['finetuned_metrics']['overall']
        deltas = comparison['deltas']
        rel_imp = comparison['relative_improvements']

        metrics_to_show = [
            ('Citation Accuracy', 'citation_accuracy_mean'),
            ('Article Accuracy', 'article_accuracy_mean'),
            ('ROUGE-L', 'rouge_l_mean'),
            ('Exact Match', 'exact_match_mean')
        ]

        print(f"\n{'Metric':<20} {'Base':<10} {'Fine-tuned':<12} {'Delta':<10} {'Rel. Imp.':<12}")
        print("-" * 80)

        for name, key in metrics_to_show:
            base_val = base.get(key, 0) * 100
            ft_val = ft.get(key, 0) * 100
            delta_val = deltas.get(key, 0) * 100
            rel_val = rel_imp.get(key, 0)

            # Color coding (simplified for console)
            indicator = "✓" if delta_val > 0 else "✗" if delta_val < 0 else "="

            print(f"{name:<20} {base_val:>6.2f}%   {ft_val:>8.2f}%   {delta_val:>+7.2f}%  {rel_val:>+9.1f}% {indicator}")

        # Win/loss/tie
        print("\n" + "-" * 80)
        print("WIN/LOSS/TIE ANALYSIS")
        print("-" * 80)

        wlt = comparison['win_loss_tie']
        print(f"\nFine-tuned wins: {wlt['finetuned_wins']}")
        print(f"Base wins:       {wlt['base_wins']}")
        print(f"Ties:            {wlt['ties']}")
        print(f"Win rate:        {wlt['finetuned_win_rate']*100:.1f}%")

        # Per-language
        print("\n" + "-" * 80)
        print("PER-LANGUAGE PERFORMANCE")
        print("-" * 80)

        base_lang = comparison['base_metrics']['by_language']
        ft_lang = comparison['finetuned_metrics']['by_language']

        print(f"\n{'Language':<12} {'Base ROUGE-L':<15} {'FT ROUGE-L':<15} {'Improvement':<12}")
        print("-" * 80)

        for lang in ['en', 'fr', 'de', 'es', 'pt']:
            if lang in base_lang and lang in ft_lang:
                base_rouge = base_lang[lang].get('rouge_l_mean', 0) * 100
                ft_rouge = ft_lang[lang].get('rouge_l_mean', 0) * 100
                improvement = ft_rouge - base_rouge

                indicator = "✓" if improvement > 0 else "✗" if improvement < 0 else "="
                print(f"{lang.upper():<12} {base_rouge:>10.2f}%    {ft_rouge:>10.2f}%    {improvement:>+9.2f}% {indicator}")

        print("\n" + "=" * 80)
        print("✓ Comparison complete!")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Compare EUR-Lex Models')
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-3.3-70B-Instruct',
                       help='Base model path')
    parser.add_argument('--finetuned_model', type=str, default='./checkpoints/sft/final',
                       help='Fine-tuned model path')
    parser.add_argument('--test_dataset', type=str, required=True,
                       help='Path to test dataset (JSONL)')
    parser.add_argument('--output_dir', type=str, default='results/model_comparison',
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for inference')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Initialize comparator
    comparator = ModelComparator(
        args.base_model,
        args.finetuned_model,
        device=args.device
    )

    # Run comparison
    comparator.run_comparison(
        args.test_dataset,
        args.output_dir,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )


if __name__ == '__main__':
    main()
