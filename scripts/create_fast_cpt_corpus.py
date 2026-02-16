"""
Create Fast CPT Corpus

Filters and selects highest-quality documents for rapid CPT training (4-6 hours).
Targets ~2B tokens (40% of full corpus) focusing on:
- Recent documents (2020+)
- Core legal areas
- Regulations and directives
- Substantive content
"""

import logging
import json
import argparse
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_quality_score(doc: Dict) -> float:
    """
    Calculate quality score for document prioritization

    Scoring criteria:
    - Recency (2020+ gets highest priority)
    - Document type (regulations > directives > others)
    - Length (longer = more substantive)
    - Subject matter (priority topics)
    - Language (ensure balanced representation)

    Args:
        doc: Parsed document dictionary

    Returns:
        Quality score (higher = better)
    """
    score = 0.0
    metadata = doc.get('metadata', {})

    # 1. Recency score (0-100 points)
    date = metadata.get('date', '')
    if date and len(date) >= 4:
        try:
            year = int(date[:4])
            if year >= 2024:
                score += 100
            elif year >= 2020:
                score += 80
            elif year >= 2015:
                score += 50
            elif year >= 2010:
                score += 20
        except ValueError:
            pass

    # 2. Document type score (0-80 points)
    doc_type = metadata.get('doc_type', '').lower()
    if doc_type == 'regulation':
        score += 80  # Regulations are binding
    elif doc_type == 'directive':
        score += 60  # Directives are important
    elif doc_type == 'decision':
        score += 40
    elif doc_type == 'recommendation':
        score += 20

    # 3. Length score (0-40 points)
    text_length = len(doc.get('full_text', ''))
    if text_length > 10000:
        score += 40  # Very substantial
    elif text_length > 5000:
        score += 30  # Substantial
    elif text_length > 2000:
        score += 20  # Adequate
    elif text_length > 500:
        score += 10  # Minimal
    # < 500 chars gets 0 points

    # 4. Subject matter score (0-30 points)
    subjects = metadata.get('subject_matter', [])
    priority_subjects = [
        'data protection', 'privacy', 'gdpr',
        'consumer protection', 'consumer rights',
        'environment', 'climate',
        'digital', 'electronic', 'cyber',
        'competition', 'antitrust',
        'financial', 'banking',
        'health', 'medical',
        'employment', 'labour'
    ]

    subject_text = ' '.join(subjects).lower()
    matches = sum(1 for ps in priority_subjects if ps in subject_text)
    score += min(matches * 10, 30)  # Cap at 30 points

    # 5. Language balance (small boost for underrepresented)
    # This ensures we don't lose minority languages
    lang = metadata.get('language', 'unknown').lower()
    if lang in ['es', 'pt']:  # Smaller languages get small boost
        score += 5

    return score


def estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation)"""
    # Average: 1.4 tokens per word, ~5 chars per word
    words = len(text) / 5
    tokens = int(words * 1.4)
    return tokens


def filter_documents_by_quality(documents: List[Dict],
                                target_tokens: int = 2_000_000_000,
                                quality_threshold: int = 70) -> List[Dict]:
    """
    Filter and select highest-quality documents

    Args:
        documents: List of parsed documents
        target_tokens: Target number of tokens (~2B for fast training)
        quality_threshold: Minimum quality score to consider

    Returns:
        Filtered list of documents
    """
    logger.info(f"Filtering {len(documents)} documents")
    logger.info(f"Target tokens: {target_tokens:,}")
    logger.info(f"Quality threshold: {quality_threshold}")

    # Score all documents
    logger.info("Calculating quality scores...")
    scored_docs = []

    for doc in tqdm(documents, desc="Scoring documents"):
        score = calculate_quality_score(doc)

        # Only consider documents above threshold
        if score >= quality_threshold:
            tokens = estimate_tokens(doc.get('full_text', ''))
            scored_docs.append((score, tokens, doc))

    logger.info(f"Documents above threshold: {len(scored_docs)}")

    # Sort by score (descending)
    scored_docs.sort(reverse=True, key=lambda x: x[0])

    # Select documents up to target token count
    selected = []
    total_tokens = 0
    language_counts = {}

    for score, tokens, doc in scored_docs:
        # Check if we've reached target
        if total_tokens >= target_tokens:
            break

        # Track language distribution
        lang = doc.get('metadata', {}).get('language', 'unknown')
        language_counts[lang] = language_counts.get(lang, 0) + 1

        selected.append(doc)
        total_tokens += tokens

    logger.info(f"Selected {len(selected)} documents")
    logger.info(f"Total tokens: {total_tokens:,}")
    logger.info(f"Language distribution: {language_counts}")

    # Calculate average score
    avg_score = np.mean([calculate_quality_score(doc) for doc in selected])
    logger.info(f"Average quality score: {avg_score:.1f}")

    return selected


def main():
    parser = argparse.ArgumentParser(description='Create fast CPT corpus')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory with parsed documents')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for filtered corpus')
    parser.add_argument('--target_tokens', type=int, default=2_000_000_000,
                       help='Target token count (default: 2B)')
    parser.add_argument('--quality_threshold', type=int, default=70,
                       help='Minimum quality score (default: 70)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    np.random.seed(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Load all documents
    logger.info(f"Loading documents from {input_dir}")
    documents = []

    json_files = list(input_dir.glob('**/*.json'))
    logger.info(f"Found {len(json_files)} JSON files")

    for json_file in tqdm(json_files, desc="Loading documents"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                doc = json.load(f)
            documents.append(doc)
        except Exception as e:
            logger.warning(f"Error loading {json_file}: {e}")

    logger.info(f"Loaded {len(documents)} documents")

    # Filter documents
    filtered_docs = filter_documents_by_quality(
        documents,
        target_tokens=args.target_tokens,
        quality_threshold=args.quality_threshold
    )

    # Save filtered documents
    logger.info(f"Saving filtered documents to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, doc in enumerate(tqdm(filtered_docs, desc="Saving documents")):
        celex = doc.get('metadata', {}).get('celex', f'doc_{i}')
        output_file = output_dir / f"{celex}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)

    # Save statistics
    stats = {
        'total_documents': len(filtered_docs),
        'estimated_tokens': sum(estimate_tokens(doc.get('full_text', ''))
                               for doc in filtered_docs),
        'language_distribution': {},
        'document_type_distribution': {},
        'average_quality_score': float(np.mean([calculate_quality_score(doc)
                                                for doc in filtered_docs]))
    }

    # Calculate distributions
    for doc in filtered_docs:
        metadata = doc.get('metadata', {})

        lang = metadata.get('language', 'unknown')
        stats['language_distribution'][lang] = \
            stats['language_distribution'].get(lang, 0) + 1

        doc_type = metadata.get('doc_type', 'unknown')
        stats['document_type_distribution'][doc_type] = \
            stats['document_type_distribution'].get(doc_type, 0) + 1

    stats_file = output_dir / 'filtering_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Statistics saved to {stats_file}")
    logger.info("Filtering complete!")

    # Display summary
    logger.info("\n" + "=" * 60)
    logger.info("FILTERING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Documents selected: {stats['total_documents']:,}")
    logger.info(f"Estimated tokens: {stats['estimated_tokens']:,}")
    logger.info(f"Average quality: {stats['average_quality_score']:.1f}")
    logger.info(f"\nLanguage distribution:")
    for lang, count in sorted(stats['language_distribution'].items()):
        pct = 100 * count / stats['total_documents']
        logger.info(f"  {lang}: {count:,} ({pct:.1f}%)")
    logger.info(f"\nDocument type distribution:")
    for dtype, count in sorted(stats['document_type_distribution'].items()):
        pct = 100 * count / stats['total_documents']
        logger.info(f"  {dtype}: {count:,} ({pct:.1f}%)")
    logger.info("=" * 60)

    logger.info("\nNext steps:")
    logger.info(f"1. Build CPT corpus from filtered documents:")
    logger.info(f"   python data_processing/dataset_builders/cpt_corpus_builder.py \\")
    logger.info(f"     --input_dir {output_dir} \\")
    logger.info(f"     --output_dir data/cpt_fast \\")
    logger.info(f"     --max_seq_length 3072 \\")
    logger.info(f"     --num_shards 16")
    logger.info(f"\n2. Train with fast config:")
    logger.info(f"   deepspeed --num_gpus=4 scripts/train_cpt.py \\")
    logger.info(f"     --config configs/cpt_config_fast.yaml")


if __name__ == '__main__':
    main()
