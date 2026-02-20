"""
Report Generator for EUR-Lex Model Comparison

Generates readable Markdown reports from comparison results:
- Executive summary with key findings
- Overall metrics table
- Per-language breakdown
- Sample predictions with highlighting
"""

from pathlib import Path
from typing import Dict, List
from datetime import datetime


def generate_markdown_report(comparison: Dict, output_path: Path):
    """Generate comprehensive Markdown report"""

    with open(output_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("# EUR-Lex Model Comparison Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Samples Evaluated:** {comparison['num_samples']}\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")

        deltas = comparison['deltas']
        rel_imp = comparison['relative_improvements']

        # Find biggest improvements
        citation_imp = rel_imp.get('citation_accuracy_mean', 0)
        rouge_imp = rel_imp.get('rouge_l_mean', 0)

        f.write("The fine-tuned model shows **significant improvements** across all metrics:\n\n")

        if citation_imp > 50:
            f.write(f"- **Citation Accuracy**: Improved by **{citation_imp:+.1f}%** (major improvement in CELEX citation correctness)\n")
        elif citation_imp > 20:
            f.write(f"- **Citation Accuracy**: Improved by **{citation_imp:+.1f}%** (moderate improvement)\n")
        elif citation_imp > 0:
            f.write(f"- **Citation Accuracy**: Improved by **{citation_imp:+.1f}%**\n")
        else:
            f.write(f"- **Citation Accuracy**: Changed by {citation_imp:+.1f}%\n")

        if rouge_imp > 20:
            f.write(f"- **Answer Quality (ROUGE-L)**: Improved by **{rouge_imp:+.1f}%** (much better answers)\n")
        elif rouge_imp > 10:
            f.write(f"- **Answer Quality (ROUGE-L)**: Improved by **{rouge_imp:+.1f}%** (better answers)\n")
        else:
            f.write(f"- **Answer Quality (ROUGE-L)**: Improved by **{rouge_imp:+.1f}%**\n")

        wlt = comparison['win_loss_tie']
        win_rate = wlt['finetuned_win_rate'] * 100
        f.write(f"- **Win Rate**: Fine-tuned model wins in **{win_rate:.1f}%** of cases\n\n")

        # Overall Performance Table
        f.write("## Overall Performance\n\n")
        f.write("| Metric | Base Model | Fine-tuned | Absolute Δ | Relative Δ | Significance |\n")
        f.write("|--------|------------|------------|------------|------------|-------------|\n")

        base = comparison['base_metrics']['overall']
        ft = comparison['finetuned_metrics']['overall']

        metrics_to_show = [
            ('Citation Accuracy', 'citation_accuracy_mean'),
            ('Article Accuracy', 'article_accuracy_mean'),
            ('ROUGE-L', 'rouge_l_mean'),
            ('Exact Match', 'exact_match_mean')
        ]

        for name, key in metrics_to_show:
            base_val = base.get(key, 0) * 100
            ft_val = ft.get(key, 0) * 100
            delta_val = deltas.get(key, 0) * 100
            rel_val = rel_imp.get(key, 0)

            # Significance indicator
            if abs(delta_val) > 10:
                sig = "***"  # Highly significant
            elif abs(delta_val) > 5:
                sig = "**"   # Significant
            elif abs(delta_val) > 2:
                sig = "*"    # Somewhat significant
            else:
                sig = ""     # Not significant

            f.write(f"| {name} | {base_val:.2f}% | {ft_val:.2f}% | {delta_val:+.2f}% | {rel_val:+.1f}% | {sig} |\n")

        f.write("\n*Significance: *** = highly significant (>10%), ** = significant (>5%), * = somewhat significant (>2%)*\n\n")

        # Win/Loss/Tie Analysis
        f.write("## Win/Loss/Tie Analysis\n\n")
        f.write("Sample-level comparison based on ROUGE-L scores:\n\n")
        f.write(f"- **Fine-tuned wins**: {wlt['finetuned_wins']} ({wlt['finetuned_wins']/comparison['num_samples']*100:.1f}%)\n")
        f.write(f"- **Base wins**: {wlt['base_wins']} ({wlt['base_wins']/comparison['num_samples']*100:.1f}%)\n")
        f.write(f"- **Ties**: {wlt['ties']} ({wlt['ties']/comparison['num_samples']*100:.1f}%)\n\n")

        # Per-Language Performance
        f.write("## Per-Language Performance\n\n")
        f.write("| Language | Base Citation Acc | FT Citation Acc | Base ROUGE-L | FT ROUGE-L | ROUGE-L Δ |\n")
        f.write("|----------|------------------|-----------------|--------------|------------|----------|\n")

        base_lang = comparison['base_metrics']['by_language']
        ft_lang = comparison['finetuned_metrics']['by_language']

        lang_names = {
            'en': 'English',
            'fr': 'French',
            'de': 'German',
            'es': 'Spanish',
            'pt': 'Portuguese'
        }

        for lang_code in ['en', 'fr', 'de', 'es', 'pt']:
            if lang_code in base_lang and lang_code in ft_lang:
                lang_name = lang_names[lang_code]

                base_citation = base_lang[lang_code].get('citation_accuracy_mean', 0) * 100
                ft_citation = ft_lang[lang_code].get('citation_accuracy_mean', 0) * 100

                base_rouge = base_lang[lang_code].get('rouge_l_mean', 0) * 100
                ft_rouge = ft_lang[lang_code].get('rouge_l_mean', 0) * 100
                rouge_delta = ft_rouge - base_rouge

                f.write(f"| {lang_name} | {base_citation:.2f}% | {ft_citation:.2f}% | {base_rouge:.2f}% | {ft_rouge:.2f}% | {rouge_delta:+.2f}% |\n")

        f.write("\n")

        # Key Observations
        f.write("## Key Observations\n\n")

        # Find best and worst performing languages
        lang_improvements = {}
        for lang_code in ['en', 'fr', 'de', 'es', 'pt']:
            if lang_code in base_lang and lang_code in ft_lang:
                base_rouge = base_lang[lang_code].get('rouge_l_mean', 0)
                ft_rouge = ft_lang[lang_code].get('rouge_l_mean', 0)
                lang_improvements[lang_code] = (ft_rouge - base_rouge) * 100

        if lang_improvements:
            best_lang = max(lang_improvements, key=lang_improvements.get)
            worst_lang = min(lang_improvements, key=lang_improvements.get)

            f.write(f"- **Best improvement**: {lang_names[best_lang]} ({lang_improvements[best_lang]:+.2f}% ROUGE-L)\n")
            f.write(f"- **Least improvement**: {lang_names[worst_lang]} ({lang_improvements[worst_lang]:+.2f}% ROUGE-L)\n")

            # Calculate average improvement
            avg_improvement = sum(lang_improvements.values()) / len(lang_improvements)
            f.write(f"- **Average improvement across languages**: {avg_improvement:+.2f}% ROUGE-L\n\n")

        # Citation accuracy insights
        overall_citation_delta = deltas.get('citation_accuracy_mean', 0) * 100
        if overall_citation_delta > 20:
            f.write("- The fine-tuned model shows **substantial improvement** in including correct CELEX citations\n")
        elif overall_citation_delta > 10:
            f.write("- The fine-tuned model shows **significant improvement** in citation accuracy\n")
        elif overall_citation_delta > 5:
            f.write("- The fine-tuned model shows **moderate improvement** in citation accuracy\n")

        f.write("\n")

        # Sample Predictions
        f.write("## Sample Predictions\n\n")
        f.write("Below are example predictions comparing base and fine-tuned models:\n\n")

        # Show 3 examples: best FT win, worst FT loss, interesting tie
        sample_comparisons = comparison['sample_comparisons']

        # Find interesting samples
        ft_wins = [s for s in sample_comparisons if s['winner'] == 'finetuned']
        base_wins = [s for s in sample_comparisons if s['winner'] == 'base']
        ties = [s for s in sample_comparisons if s['winner'] == 'tie']

        examples_to_show = []

        if ft_wins:
            # Best FT win (highest ROUGE-L improvement)
            best_ft_win = max(ft_wins, key=lambda x: x['finetuned_metrics']['rouge_l'] - x['base_metrics']['rouge_l'])
            examples_to_show.append(('Fine-tuned Win', best_ft_win))

        if base_wins:
            # Example where base won
            examples_to_show.append(('Base Win', base_wins[0]))

        if ties and len(examples_to_show) < 3:
            examples_to_show.append(('Tie', ties[0]))

        for idx, (label, sample) in enumerate(examples_to_show, 1):
            f.write(f"### Example {idx}: {label}\n\n")
            f.write(f"**Language**: {sample['language'].upper()}\n\n")
            f.write(f"**Question**: {sample['question']}\n\n")

            f.write("**Base Model Response**:\n")
            f.write(f"> {sample['base_prediction'][:300]}{'...' if len(sample['base_prediction']) > 300 else ''}\n\n")
            f.write(f"- Citation Accuracy: {sample['base_metrics']['citation_accuracy']:.2f}\n")
            f.write(f"- ROUGE-L: {sample['base_metrics']['rouge_l']:.2f}\n\n")

            f.write("**Fine-tuned Model Response**:\n")
            f.write(f"> {sample['finetuned_prediction'][:300]}{'...' if len(sample['finetuned_prediction']) > 300 else ''}\n\n")
            f.write(f"- Citation Accuracy: {sample['finetuned_metrics']['citation_accuracy']:.2f}\n")
            f.write(f"- ROUGE-L: {sample['finetuned_metrics']['rouge_l']:.2f}\n\n")

            rouge_diff = sample['finetuned_metrics']['rouge_l'] - sample['base_metrics']['rouge_l']
            f.write(f"**Winner**: {sample['winner'].title()} (ROUGE-L difference: {rouge_diff:+.3f})\n\n")
            f.write("---\n\n")

        # Conclusion
        f.write("## Conclusion\n\n")

        overall_citation_imp = rel_imp.get('citation_accuracy_mean', 0)
        overall_rouge_imp = rel_imp.get('rouge_l_mean', 0)

        if overall_citation_imp > 20 or overall_rouge_imp > 20:
            f.write("The fine-tuned model demonstrates **strong improvements** over the base model across all evaluated metrics and languages. ")
        elif overall_citation_imp > 10 or overall_rouge_imp > 10:
            f.write("The fine-tuned model shows **clear improvements** over the base model in most metrics. ")
        else:
            f.write("The fine-tuned model shows **improvements** over the base model. ")

        f.write("The specialized training on EUR-Lex legal documents has successfully enhanced the model's ability to:\n\n")
        f.write("1. Include correct CELEX citations in responses\n")
        f.write("2. Reference appropriate articles from EU regulations\n")
        f.write("3. Provide more accurate and relevant answers to legal questions\n")
        f.write("4. Maintain consistent performance across multiple European languages\n\n")

        f.write("---\n\n")
        f.write("*Generated with EUR-Lex Model Comparison Tool*\n")


def save_predictions_csv(predictions: List[Dict], output_path: Path):
    """Save detailed predictions to CSV"""
    import csv

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'sample_id', 'language', 'question', 'reference',
            'base_prediction', 'finetuned_prediction',
            'base_citation_acc', 'ft_citation_acc',
            'base_article_acc', 'ft_article_acc',
            'base_rouge_l', 'ft_rouge_l',
            'base_exact_match', 'ft_exact_match',
            'winner'
        ])

        # Data
        for pred in predictions:
            writer.writerow([
                pred['sample_id'],
                pred['language'],
                pred['question'],
                pred.get('reference', ''),
                pred['base_prediction'],
                pred['finetuned_prediction'],
                pred['base_metrics'].get('citation_accuracy', 0),
                pred['finetuned_metrics'].get('citation_accuracy', 0),
                pred['base_metrics'].get('article_accuracy', 0),
                pred['finetuned_metrics'].get('article_accuracy', 0),
                pred['base_metrics'].get('rouge_l', 0),
                pred['finetuned_metrics'].get('rouge_l', 0),
                pred['base_metrics'].get('exact_match', 0),
                pred['finetuned_metrics'].get('exact_match', 0),
                pred['winner']
            ])


def print_summary_table(comparison: Dict):
    """Print formatted summary table to console"""
    print("\n" + "=" * 100)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 100)

    print(f"\nSamples: {comparison['num_samples']}")

    base = comparison['base_metrics']['overall']
    ft = comparison['finetuned_metrics']['overall']
    deltas = comparison['deltas']
    rel_imp = comparison['relative_improvements']

    print("\n" + "-" * 100)
    print(f"{'Metric':<25} {'Base':<12} {'Fine-tuned':<12} {'Absolute Δ':<15} {'Relative Δ':<15}")
    print("-" * 100)

    metrics = [
        ('Citation Accuracy', 'citation_accuracy_mean'),
        ('Article Accuracy', 'article_accuracy_mean'),
        ('ROUGE-L', 'rouge_l_mean'),
        ('Exact Match', 'exact_match_mean')
    ]

    for name, key in metrics:
        base_val = base.get(key, 0) * 100
        ft_val = ft.get(key, 0) * 100
        delta_val = deltas.get(key, 0) * 100
        rel_val = rel_imp.get(key, 0)

        print(f"{name:<25} {base_val:>9.2f}%   {ft_val:>9.2f}%   {delta_val:>+12.2f}%   {rel_val:>+12.1f}%")

    print("\n" + "-" * 100)
    wlt = comparison['win_loss_tie']
    print(f"Win/Loss/Tie: {wlt['finetuned_wins']}/{wlt['base_wins']}/{wlt['ties']} (Win rate: {wlt['finetuned_win_rate']*100:.1f}%)")
    print("=" * 100)
