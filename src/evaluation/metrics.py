"""
Evaluation Metrics for Legal Q&A

Includes:
- Citation Accuracy: Verify CELEX numbers and article references
- ROUGE-L: Answer similarity
- Exact Match: For factual questions
- Hallucination Detection: Identify incorrect citations
"""

import re
from typing import List, Dict, Tuple
import numpy as np
from rouge_score import rouge_scorer


class LegalQAMetrics:
    """Metrics for evaluating legal Q&A model performance"""

    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def citation_accuracy(self, generated: str, reference: str) -> float:
        """
        Check if generated answer contains correct CELEX citations

        Args:
            generated: Generated answer
            reference: Reference answer

        Returns:
            1.0 if citations match, 0.0 otherwise
        """
        # Extract CELEX numbers from both
        ref_celex = self._extract_celex(reference)
        gen_celex = self._extract_celex(generated)

        if not ref_celex:
            # No citation required
            return 1.0

        if not gen_celex:
            # Citation required but missing
            return 0.0

        # Check if any reference CELEX appears in generated
        for celex in ref_celex:
            if celex in gen_celex:
                return 1.0

        return 0.0

    def article_accuracy(self, generated: str, reference: str) -> float:
        """
        Check if generated answer cites correct articles

        Args:
            generated: Generated answer
            reference: Reference answer

        Returns:
            1.0 if articles match, 0.0 otherwise
        """
        ref_articles = self._extract_articles(reference)
        gen_articles = self._extract_articles(generated)

        if not ref_articles:
            return 1.0

        if not gen_articles:
            return 0.0

        # Check overlap
        overlap = len(set(ref_articles) & set(gen_articles))
        return overlap / len(ref_articles)

    def rouge_l(self, generated: str, reference: str) -> float:
        """
        Compute ROUGE-L F1 score

        Args:
            generated: Generated answer
            reference: Reference answer

        Returns:
            ROUGE-L F1 score
        """
        scores = self.rouge_scorer.score(reference, generated)
        return scores['rougeL'].fmeasure

    def exact_match(self, generated: str, reference: str) -> float:
        """
        Check for exact match (normalized)

        Args:
            generated: Generated answer
            reference: Reference answer

        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        gen_norm = self._normalize_text(generated)
        ref_norm = self._normalize_text(reference)

        return 1.0 if gen_norm == ref_norm else 0.0

    def citation_hallucination_rate(self, generated: str, valid_celex: List[str]) -> float:
        """
        Detect if generated citations are hallucinated (not in valid set)

        Args:
            generated: Generated answer
            valid_celex: List of valid CELEX numbers

        Returns:
            Proportion of hallucinated citations (0.0 = none, 1.0 = all)
        """
        gen_celex = self._extract_celex(generated)

        if not gen_celex:
            return 0.0

        hallucinated = sum(1 for celex in gen_celex if celex not in valid_celex)
        return hallucinated / len(gen_celex)

    def _extract_celex(self, text: str) -> List[str]:
        """Extract CELEX numbers from text"""
        # CELEX format: 32016R0679 (year + type + number)
        pattern = r'CELEX:\s*(\d{5}[A-Z]\d{4})|(\d{5}[A-Z]\d{4})'
        matches = re.findall(pattern, text, re.IGNORECASE)

        # Flatten tuples and remove empty strings
        celex_numbers = []
        for match in matches:
            for group in match:
                if group:
                    celex_numbers.append(group.upper())

        return celex_numbers

    def _extract_articles(self, text: str) -> List[str]:
        """Extract article numbers from text"""
        pattern = r'Article\s+(\d+[a-z]?)'
        matches = re.findall(pattern, text, re.IGNORECASE)
        return [m.lower() for m in matches]

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        # Lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()

    def compute_all_metrics(self, generated: str, reference: str,
                           valid_celex: List[str] = None) -> Dict[str, float]:
        """
        Compute all metrics for a single example

        Args:
            generated: Generated answer
            reference: Reference answer
            valid_celex: List of valid CELEX numbers (for hallucination detection)

        Returns:
            Dictionary of metric scores
        """
        metrics = {
            'citation_accuracy': self.citation_accuracy(generated, reference),
            'article_accuracy': self.article_accuracy(generated, reference),
            'rouge_l': self.rouge_l(generated, reference),
            'exact_match': self.exact_match(generated, reference),
        }

        if valid_celex is not None:
            metrics['hallucination_rate'] = self.citation_hallucination_rate(
                generated, valid_celex
            )

        return metrics

    def aggregate_metrics(self, all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Aggregate metrics across multiple examples

        Args:
            all_metrics: List of metric dictionaries

        Returns:
            Dictionary of aggregated metrics
        """
        if not all_metrics:
            return {}

        aggregated = {}

        # Get all metric names
        metric_names = all_metrics[0].keys()

        for metric_name in metric_names:
            values = [m[metric_name] for m in all_metrics if metric_name in m]
            aggregated[f'{metric_name}_mean'] = np.mean(values)
            aggregated[f'{metric_name}_std'] = np.std(values)

        return aggregated


class LanguageSpecificMetrics:
    """Track metrics per language"""

    def __init__(self):
        self.metrics_by_language = {
            'en': [],
            'fr': [],
            'de': [],
            'es': [],
            'pt': []
        }
        self.base_metrics = LegalQAMetrics()

    def add_example(self, generated: str, reference: str, language: str,
                   valid_celex: List[str] = None):
        """Add example to language-specific tracking"""
        metrics = self.base_metrics.compute_all_metrics(generated, reference, valid_celex)

        if language in self.metrics_by_language:
            self.metrics_by_language[language].append(metrics)

    def get_language_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics per language"""
        summary = {}

        for lang, metrics_list in self.metrics_by_language.items():
            if not metrics_list:
                continue

            summary[lang] = self.base_metrics.aggregate_metrics(metrics_list)

        return summary


def test_metrics():
    """Test metrics with sample data"""
    metrics = LegalQAMetrics()

    # Test 1: Perfect match
    reference = "According to Article 5, GDPR (CELEX: 32016R0679), personal data must be processed lawfully."
    generated = "According to Article 5, GDPR (CELEX: 32016R0679), personal data must be processed lawfully."

    scores = metrics.compute_all_metrics(generated, reference)
    print("Test 1 - Perfect match:")
    for key, value in scores.items():
        print(f"  {key}: {value:.3f}")

    # Test 2: Correct citation, different wording
    reference = "According to Article 5, GDPR (CELEX: 32016R0679), personal data must be processed lawfully."
    generated = "Article 5 of the GDPR (CELEX: 32016R0679) states that data processing must be lawful."

    scores = metrics.compute_all_metrics(generated, reference)
    print("\nTest 2 - Correct citation, different wording:")
    for key, value in scores.items():
        print(f"  {key}: {value:.3f}")

    # Test 3: Wrong article
    reference = "According to Article 5, GDPR (CELEX: 32016R0679), personal data must be processed lawfully."
    generated = "According to Article 6, GDPR (CELEX: 32016R0679), personal data must be processed lawfully."

    scores = metrics.compute_all_metrics(generated, reference)
    print("\nTest 3 - Wrong article:")
    for key, value in scores.items():
        print(f"  {key}: {value:.3f}")

    # Test 4: Missing citation
    reference = "According to Article 5, GDPR (CELEX: 32016R0679), personal data must be processed lawfully."
    generated = "Personal data must be processed lawfully according to GDPR."

    scores = metrics.compute_all_metrics(generated, reference)
    print("\nTest 4 - Missing citation:")
    for key, value in scores.items():
        print(f"  {key}: {value:.3f}")


if __name__ == '__main__':
    print("Testing Legal Q&A Metrics...")
    print("=" * 60)
    test_metrics()
