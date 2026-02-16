"""
CPT Corpus Builder for EUR-Lex Legal Documents

Builds Continued Pretraining (CPT) corpus from parsed FORMEX documents:
- Applies language distribution (EN 35%, FR 25%, DE 20%, ES 12%, PT 8%)
- Adds language tags
- Packs documents into fixed-length sequences (4096 tokens)
- Creates sharded datasets for distributed training
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import Dataset
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CPTCorpusConfig:
    """Configuration for CPT corpus building"""
    max_seq_length: int = 4096
    num_shards: int = 32
    language_distribution: Dict[str, float] = None
    add_language_tags: bool = True
    min_doc_length: int = 100  # Minimum document length in characters
    validation_split: float = 0.05  # 5% for validation

    def __post_init__(self):
        if self.language_distribution is None:
            self.language_distribution = {
                'en': 0.35,
                'fr': 0.25,
                'de': 0.20,
                'es': 0.12,
                'pt': 0.08
            }


class CPTCorpusBuilder:
    """Builder for CPT training corpus"""

    def __init__(self, config: CPTCorpusConfig, tokenizer_name: str = "meta-llama/Llama-3.1-70B-Instruct"):
        """
        Initialize CPT corpus builder

        Args:
            config: Corpus building configuration
            tokenizer_name: HuggingFace tokenizer to use
        """
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.sep_token = "<|doc_sep|>"  # Document separator token

        # Add special separator token if not in vocabulary
        if self.sep_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([self.sep_token])
            logger.info(f"Added separator token: {self.sep_token}")

    def load_parsed_documents(self, input_dir: Path) -> Dict[str, List[Dict]]:
        """
        Load parsed documents organized by language

        Args:
            input_dir: Directory containing parsed JSON files

        Returns:
            Dictionary mapping language code to list of documents
        """
        logger.info(f"Loading parsed documents from {input_dir}")

        documents_by_lang = {
            'en': [],
            'fr': [],
            'de': [],
            'es': [],
            'pt': []
        }

        json_files = list(input_dir.glob('**/*.json'))
        logger.info(f"Found {len(json_files)} JSON files")

        for json_file in tqdm(json_files, desc="Loading documents"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    doc = json.load(f)

                # Extract language and text
                lang = doc.get('metadata', {}).get('language', 'unknown')
                full_text = doc.get('full_text', '')

                # Filter out short documents
                if len(full_text) < self.config.min_doc_length:
                    continue

                # Only keep languages we're interested in
                if lang in documents_by_lang:
                    documents_by_lang[lang].append(doc)

            except Exception as e:
                logger.warning(f"Error loading {json_file}: {e}")
                continue

        # Log statistics
        for lang, docs in documents_by_lang.items():
            logger.info(f"Loaded {len(docs)} documents for language: {lang}")

        return documents_by_lang

    def sample_by_distribution(self, documents_by_lang: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Sample documents according to target language distribution

        Args:
            documents_by_lang: Documents organized by language

        Returns:
            List of sampled documents
        """
        logger.info("Sampling documents by target distribution")

        # Calculate total number of documents
        total_docs = sum(len(docs) for docs in documents_by_lang.values())
        logger.info(f"Total documents: {total_docs}")

        sampled_documents = []

        for lang, target_ratio in self.config.language_distribution.items():
            if lang not in documents_by_lang or len(documents_by_lang[lang]) == 0:
                logger.warning(f"No documents found for language: {lang}")
                continue

            # Calculate target number of documents for this language
            target_count = int(total_docs * target_ratio)
            available_count = len(documents_by_lang[lang])

            # If we don't have enough documents, use all available and warn
            if available_count < target_count:
                logger.warning(
                    f"Insufficient documents for {lang}: "
                    f"need {target_count}, have {available_count}"
                )
                sample_count = available_count
            else:
                sample_count = target_count

            # Sample documents
            if sample_count >= available_count:
                sampled = documents_by_lang[lang]
            else:
                indices = np.random.choice(available_count, sample_count, replace=False)
                sampled = [documents_by_lang[lang][i] for i in indices]

            sampled_documents.extend(sampled)
            logger.info(f"Sampled {len(sampled)} documents for {lang}")

        # Shuffle all documents
        np.random.shuffle(sampled_documents)

        logger.info(f"Total sampled documents: {len(sampled_documents)}")
        return sampled_documents

    def prepare_text_for_cpt(self, doc: Dict) -> str:
        """
        Prepare document text for CPT training

        Args:
            doc: Parsed document

        Returns:
            Formatted text with language tag
        """
        lang = doc.get('metadata', {}).get('language', 'unknown').upper()
        text = doc.get('full_text', '')

        if self.config.add_language_tags:
            return f"[{lang}] {text}"
        else:
            return text

    def pack_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Pack documents into fixed-length sequences

        Args:
            documents: List of documents

        Returns:
            List of packed sequences with metadata
        """
        logger.info(f"Packing {len(documents)} documents into sequences")

        packed_sequences = []
        current_sequence = []
        current_tokens = []
        current_length = 0

        for doc in tqdm(documents, desc="Packing documents"):
            # Prepare text
            text = self.prepare_text_for_cpt(doc)

            # Tokenize
            tokens = self.tokenizer.encode(text, add_special_tokens=False)

            # Check if adding this document would exceed max length
            # +1 for separator token
            if current_length + len(tokens) + 1 <= self.config.max_seq_length:
                # Add to current sequence
                current_sequence.append(doc)
                current_tokens.extend(tokens)
                current_tokens.append(self.tokenizer.convert_tokens_to_ids(self.sep_token))
                current_length += len(tokens) + 1
            else:
                # Save current sequence if it has content
                if current_sequence:
                    packed_sequences.append({
                        'input_ids': current_tokens,
                        'num_documents': len(current_sequence),
                        'length': current_length
                    })

                # Start new sequence with current document
                current_sequence = [doc]
                current_tokens = tokens + [self.tokenizer.convert_tokens_to_ids(self.sep_token)]
                current_length = len(tokens) + 1

        # Add last sequence
        if current_sequence:
            packed_sequences.append({
                'input_ids': current_tokens,
                'num_documents': len(current_sequence),
                'length': current_length
            })

        logger.info(f"Created {len(packed_sequences)} packed sequences")

        # Log statistics
        lengths = [seq['length'] for seq in packed_sequences]
        logger.info(f"Sequence length stats: "
                   f"mean={np.mean(lengths):.0f}, "
                   f"min={np.min(lengths)}, "
                   f"max={np.max(lengths)}")

        return packed_sequences

    def create_shards(self, sequences: List[Dict], output_dir: Path, split: str = "train"):
        """
        Create sharded Parquet files for distributed training

        Args:
            sequences: List of packed sequences
            output_dir: Output directory
            split: Dataset split (train/validation)
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Calculate sequences per shard
        sequences_per_shard = len(sequences) // self.config.num_shards
        if sequences_per_shard == 0:
            sequences_per_shard = 1

        logger.info(f"Creating {self.config.num_shards} shards with ~{sequences_per_shard} sequences each")

        for shard_idx in range(self.config.num_shards):
            start_idx = shard_idx * sequences_per_shard
            end_idx = start_idx + sequences_per_shard if shard_idx < self.config.num_shards - 1 else len(sequences)

            shard_sequences = sequences[start_idx:end_idx]

            if not shard_sequences:
                continue

            # Convert to HuggingFace Dataset
            dataset = Dataset.from_list(shard_sequences)

            # Save as Parquet
            shard_filename = f"cpt_{split}_shard_{shard_idx:02d}.parquet"
            shard_path = output_dir / shard_filename

            dataset.to_parquet(str(shard_path))
            logger.info(f"Saved shard {shard_idx}: {shard_path} ({len(shard_sequences)} sequences)")

        logger.info(f"Successfully created {self.config.num_shards} shards")

    def build_corpus(self, input_dir: Path, output_dir: Path):
        """
        Build complete CPT corpus

        Args:
            input_dir: Directory containing parsed documents
            output_dir: Output directory for corpus
        """
        logger.info("Starting CPT corpus building")

        # Load documents
        documents_by_lang = self.load_parsed_documents(input_dir)

        # Sample by distribution
        documents = self.sample_by_distribution(documents_by_lang)

        if not documents:
            logger.error("No documents to process")
            return

        # Split into train/validation
        np.random.shuffle(documents)
        split_idx = int(len(documents) * (1 - self.config.validation_split))
        train_documents = documents[:split_idx]
        val_documents = documents[split_idx:]

        logger.info(f"Train documents: {len(train_documents)}")
        logger.info(f"Validation documents: {len(val_documents)}")

        # Pack documents
        train_sequences = self.pack_documents(train_documents)
        val_sequences = self.pack_documents(val_documents)

        # Create shards
        logger.info("Creating training shards")
        self.create_shards(train_sequences, output_dir / "train", split="train")

        logger.info("Creating validation shards")
        # Validation typically doesn't need multiple shards
        val_output_dir = output_dir / "validation"
        val_output_dir.mkdir(parents=True, exist_ok=True)

        val_dataset = Dataset.from_list(val_sequences)
        val_dataset.to_parquet(str(val_output_dir / "cpt_val.parquet"))

        logger.info(f"CPT corpus building complete!")

        # Generate statistics report
        self.generate_statistics_report(train_sequences, val_sequences, output_dir)

    def generate_statistics_report(self, train_sequences: List[Dict], val_sequences: List[Dict], output_dir: Path):
        """Generate corpus statistics report"""
        stats = {
            'train': {
                'num_sequences': len(train_sequences),
                'total_tokens': sum(seq['length'] for seq in train_sequences),
                'avg_length': np.mean([seq['length'] for seq in train_sequences]),
                'total_documents': sum(seq['num_documents'] for seq in train_sequences)
            },
            'validation': {
                'num_sequences': len(val_sequences),
                'total_tokens': sum(seq['length'] for seq in val_sequences),
                'avg_length': np.mean([seq['length'] for seq in val_sequences]),
                'total_documents': sum(seq['num_documents'] for seq in val_sequences)
            }
        }

        report_path = output_dir / "corpus_statistics.json"
        with open(report_path, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Statistics report saved to {report_path}")
        logger.info(f"Train: {stats['train']['total_tokens']:,} tokens, {stats['train']['num_sequences']:,} sequences")
        logger.info(f"Validation: {stats['validation']['total_tokens']:,} tokens, {stats['validation']['num_sequences']:,} sequences")


def main():
    """Command-line interface"""
    import argparse

    parser = argparse.ArgumentParser(description='Build CPT corpus from parsed documents')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing parsed JSON documents')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for CPT corpus')
    parser.add_argument('--max_seq_length', type=int, default=4096,
                       help='Maximum sequence length')
    parser.add_argument('--num_shards', type=int, default=32,
                       help='Number of training shards')
    parser.add_argument('--tokenizer', type=str, default="meta-llama/Llama-3.1-70B-Instruct",
                       help='Tokenizer to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Create configuration
    config = CPTCorpusConfig(
        max_seq_length=args.max_seq_length,
        num_shards=args.num_shards
    )

    # Build corpus
    builder = CPTCorpusBuilder(config, tokenizer_name=args.tokenizer)
    builder.build_corpus(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir)
    )


if __name__ == '__main__':
    main()
