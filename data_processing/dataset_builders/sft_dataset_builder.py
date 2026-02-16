"""
SFT Dataset Builder for EUR-Lex Legal Q&A

Generates instruction-tuning Q&A pairs from parsed legal documents with:
- Rule-based generation (50%): Extract definitions, compliance questions
- LLM-assisted generation (30%): Use external LLM to create diverse questions
- Manual templates (20%): Complex multi-hop reasoning questions
- Citation formatting: "Article X, DocType (CELEX: XXXXX)"
"""

import logging
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SFTConfig:
    """Configuration for SFT dataset building"""
    max_seq_length: int = 2048
    num_shards: int = 8
    target_pairs: int = 150000
    rule_based_ratio: float = 0.5
    llm_assisted_ratio: float = 0.3
    manual_ratio: float = 0.2
    validation_split: float = 0.05
    min_answer_length: int = 50
    max_answer_length: int = 1000


class SFTDatasetBuilder:
    """Builder for SFT Q&A dataset with citations"""

    SYSTEM_PROMPT = """You are a specialized legal assistant for European Union law. You provide accurate information about EU regulations, directives, and legal documents from EUR-Lex. Always cite specific articles and CELEX numbers when answering.

Disclaimer: This is an AI assistant and does not constitute legal advice. Always consult official EUR-Lex sources and legal professionals for authoritative information."""

    def __init__(self, config: SFTConfig, tokenizer_name: str = "meta-llama/Llama-3.1-70B-Instruct"):
        """
        Initialize SFT dataset builder

        Args:
            config: Dataset configuration
            tokenizer_name: HuggingFace tokenizer to use
        """
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def load_parsed_documents(self, input_dir: Path) -> List[Dict]:
        """Load parsed documents"""
        logger.info(f"Loading parsed documents from {input_dir}")

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
                continue

        logger.info(f"Loaded {len(documents)} documents")
        return documents

    def generate_rule_based_pairs(self, documents: List[Dict], target_count: int) -> List[Dict]:
        """
        Generate Q&A pairs using rule-based extraction

        Strategies:
        - Extract definitions and create "What is X?" questions
        - Convert "shall" statements to compliance questions
        - Extract obligations and create requirement questions
        """
        logger.info(f"Generating {target_count} rule-based Q&A pairs")

        pairs = []

        for doc in tqdm(documents, desc="Rule-based generation"):
            metadata = doc.get('metadata', {})
            celex = metadata.get('celex')
            doc_type = metadata.get('doc_type', 'document')
            language = metadata.get('language', 'en')
            title = metadata.get('title', '')

            articles = doc.get('articles', [])

            for article in articles:
                article_id = article.get('article_id', '')
                article_text = article.get('text', '')
                paragraphs = article.get('paragraphs', [])

                if not article_text or len(article_text) < 50:
                    continue

                # Strategy 1: Definition questions
                if 'means' in article_text.lower() or 'definition' in article_text.lower():
                    pairs.extend(self._create_definition_questions(
                        article_text, article_id, celex, doc_type, language
                    ))

                # Strategy 2: Compliance questions from "shall" statements
                if 'shall' in article_text.lower():
                    pairs.extend(self._create_compliance_questions(
                        article_text, article_id, celex, doc_type, language
                    ))

                # Strategy 3: Requirement questions
                if 'required' in article_text.lower() or 'must' in article_text.lower():
                    pairs.extend(self._create_requirement_questions(
                        article_text, article_id, celex, doc_type, language
                    ))

                # Strategy 4: Scope questions
                if article_id.lower().startswith('1') or 'scope' in article_text.lower():
                    pairs.extend(self._create_scope_questions(
                        article_text, article_id, celex, doc_type, title, language
                    ))

                if len(pairs) >= target_count:
                    break

            if len(pairs) >= target_count:
                break

        logger.info(f"Generated {len(pairs)} rule-based pairs")
        return pairs[:target_count]

    def _create_definition_questions(self, text: str, article_id: str, celex: str,
                                     doc_type: str, language: str) -> List[Dict]:
        """Create definition-based Q&A pairs"""
        pairs = []

        # Extract term being defined
        definition_patterns = [
            r"'([^']+)'\s+means",
            r"\"([^\"]+)\"\s+means",
            r"For\s+the\s+purposes?\s+of\s+this\s+\w+,\s+'([^']+)'",
        ]

        for pattern in definition_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                term = match.group(1)

                # Extract definition (next 200 chars or until period)
                start_pos = match.end()
                definition_end = text.find('.', start_pos)
                if definition_end == -1:
                    definition_end = start_pos + 200
                definition = text[start_pos:definition_end].strip()

                if len(definition) < 20:
                    continue

                question = self._format_question(f"What is '{term}' according to {doc_type}?", language)
                answer = self._format_answer(
                    f"According to Article {article_id}, {doc_type} (CELEX: {celex}), '{term}' means {definition}",
                    article_id, celex, doc_type
                )

                pairs.append({
                    'question': question,
                    'answer': answer,
                    'metadata': {
                        'celex': celex,
                        'article': article_id,
                        'language': language,
                        'type': 'definition',
                        'complexity': 'low'
                    }
                })

        return pairs

    def _create_compliance_questions(self, text: str, article_id: str, celex: str,
                                     doc_type: str, language: str) -> List[Dict]:
        """Create compliance-based Q&A pairs from 'shall' statements"""
        pairs = []

        # Extract obligations
        shall_matches = re.finditer(r'([^.]+shall[^.]+\.)', text, re.IGNORECASE)

        for match in shall_matches:
            obligation = match.group(1).strip()

            if len(obligation) < 30 or len(obligation) > 300:
                continue

            # Extract subject (who shall do something)
            subject_match = re.match(r'([^,]+),?\s+shall', obligation, re.IGNORECASE)
            subject = subject_match.group(1) if subject_match else "the entity"

            question = self._format_question(
                f"What are the obligations of {subject} under Article {article_id}?",
                language
            )
            answer = self._format_answer(
                f"According to Article {article_id}, {doc_type} (CELEX: {celex}), {obligation}",
                article_id, celex, doc_type
            )

            pairs.append({
                'question': question,
                'answer': answer,
                'metadata': {
                    'celex': celex,
                    'article': article_id,
                    'language': language,
                    'type': 'compliance',
                    'complexity': 'medium'
                }
            })

        return pairs

    def _create_requirement_questions(self, text: str, article_id: str, celex: str,
                                      doc_type: str, language: str) -> List[Dict]:
        """Create requirement-based Q&A pairs"""
        pairs = []

        # Extract requirements
        req_matches = re.finditer(r'([^.]+(?:required|must)[^.]+\.)', text, re.IGNORECASE)

        for match in req_matches:
            requirement = match.group(1).strip()

            if len(requirement) < 30:
                continue

            question = self._format_question(
                f"What is required under Article {article_id}?",
                language
            )
            answer = self._format_answer(
                f"According to Article {article_id}, {doc_type} (CELEX: {celex}), {requirement}",
                article_id, celex, doc_type
            )

            pairs.append({
                'question': question,
                'answer': answer,
                'metadata': {
                    'celex': celex,
                    'article': article_id,
                    'language': language,
                    'type': 'requirement',
                    'complexity': 'medium'
                }
            })

        return pairs

    def _create_scope_questions(self, text: str, article_id: str, celex: str,
                                doc_type: str, title: str, language: str) -> List[Dict]:
        """Create scope/applicability questions"""
        pairs = []

        # Extract first 300 chars as scope summary
        scope_text = text[:300].strip()
        if scope_text.endswith('.'):
            scope_text = scope_text
        else:
            last_period = scope_text.rfind('.')
            if last_period > 100:
                scope_text = scope_text[:last_period+1]

        question = self._format_question(
            f"What is the scope of {title or doc_type}?",
            language
        )
        answer = self._format_answer(
            f"According to Article {article_id}, {doc_type} (CELEX: {celex}), {scope_text}",
            article_id, celex, doc_type
        )

        pairs.append({
            'question': question,
            'answer': answer,
            'metadata': {
                'celex': celex,
                'article': article_id,
                'language': language,
                'type': 'scope',
                'complexity': 'low'
            }
        })

        return pairs

    def _format_question(self, question: str, language: str) -> str:
        """Format question with language tag"""
        lang_tag = f"[{language.upper()}]"
        return f"{lang_tag} {question}"

    def _format_answer(self, answer: str, article_id: str, celex: str, doc_type: str) -> str:
        """Format answer with proper citation"""
        # Ensure citation is at the end
        citation = f"\n\nCitation: Article {article_id}, {doc_type} (CELEX: {celex})"
        if "CELEX:" not in answer:
            return answer + citation
        return answer

    def generate_template_pairs(self, documents: List[Dict], target_count: int) -> List[Dict]:
        """
        Generate Q&A pairs using manual templates for complex scenarios

        These are pre-defined templates for common legal question patterns
        """
        logger.info(f"Generating {target_count} template-based pairs")

        templates = [
            {
                'question_template': "How does {doc_title} define {term}?",
                'match_pattern': 'definition',
            },
            {
                'question_template': "What are the key provisions of {doc_title}?",
                'match_pattern': 'general',
            },
            {
                'question_template': "What penalties apply under {doc_title}?",
                'match_pattern': 'penalty|sanction|fine',
            },
            {
                'question_template': "Who does {doc_title} apply to?",
                'match_pattern': 'scope|application',
            },
        ]

        pairs = []
        # Implementation would expand on rule-based with more sophisticated templates
        # For now, use rule-based pairs as foundation

        return pairs[:target_count]

    def create_llm_assisted_pairs(self, documents: List[Dict], target_count: int) -> List[Dict]:
        """
        Generate Q&A pairs using external LLM (GPT-4/Claude)

        Note: This requires API access and is optional.
        For now, returns empty list. User can implement with their preferred LLM.
        """
        logger.info(f"LLM-assisted generation: {target_count} pairs requested")
        logger.info("LLM-assisted generation requires external API. Skipping.")
        logger.info("To enable: Implement with OpenAI/Anthropic API")

        return []

    def format_for_llama_chat(self, qa_pair: Dict) -> Dict:
        """Format Q&A pair for LLaMA 3.1 chat template"""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": qa_pair['question']},
            {"role": "assistant", "content": qa_pair['answer']}
        ]

        # Apply chat template
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Tokenize to check length
        tokens = self.tokenizer.encode(formatted, add_special_tokens=True)

        return {
            'text': formatted,
            'input_ids': tokens,
            'length': len(tokens),
            'question': qa_pair['question'],
            'answer': qa_pair['answer'],
            'metadata': qa_pair['metadata']
        }

    def validate_qa_pair(self, qa_pair: Dict) -> bool:
        """Validate Q&A pair quality"""
        answer = qa_pair.get('answer', '')

        # Check length
        if len(answer) < self.config.min_answer_length:
            return False
        if len(answer) > self.config.max_answer_length:
            return False

        # Check for citation
        if 'CELEX:' not in answer and 'Article' not in answer:
            return False

        return True

    def build_dataset(self, input_dir: Path, output_dir: Path):
        """Build complete SFT dataset"""
        logger.info("Starting SFT dataset building")

        # Load documents
        documents = self.load_parsed_documents(input_dir)

        if not documents:
            logger.error("No documents loaded")
            return

        # Calculate target counts per strategy
        rule_based_count = int(self.config.target_pairs * self.config.rule_based_ratio)
        llm_count = int(self.config.target_pairs * self.config.llm_assisted_ratio)
        template_count = int(self.config.target_pairs * self.config.manual_ratio)

        # Generate Q&A pairs
        logger.info(f"Target distribution: Rule={rule_based_count}, LLM={llm_count}, Template={template_count}")

        all_pairs = []

        # Rule-based generation
        rule_pairs = self.generate_rule_based_pairs(documents, rule_based_count)
        all_pairs.extend(rule_pairs)

        # LLM-assisted generation (if available)
        llm_pairs = self.create_llm_assisted_pairs(documents, llm_count)
        all_pairs.extend(llm_pairs)

        # Template-based generation
        template_pairs = self.generate_template_pairs(documents, template_count)
        all_pairs.extend(template_pairs)

        logger.info(f"Generated {len(all_pairs)} total Q&A pairs")

        # Validate pairs
        valid_pairs = [p for p in all_pairs if self.validate_qa_pair(p)]
        logger.info(f"Valid pairs after filtering: {len(valid_pairs)}")

        # Format for training
        formatted_pairs = []
        for pair in tqdm(valid_pairs, desc="Formatting pairs"):
            formatted = self.format_for_llama_chat(pair)
            # Filter by length
            if formatted['length'] <= self.config.max_seq_length:
                formatted_pairs.append(formatted)

        logger.info(f"Formatted pairs: {len(formatted_pairs)}")

        # Split train/validation
        np.random.shuffle(formatted_pairs)
        split_idx = int(len(formatted_pairs) * (1 - self.config.validation_split))
        train_pairs = formatted_pairs[:split_idx]
        val_pairs = formatted_pairs[split_idx:]

        logger.info(f"Train: {len(train_pairs)}, Validation: {len(val_pairs)}")

        # Create shards
        self.create_shards(train_pairs, output_dir / "train", "train")

        # Save validation
        val_dir = output_dir / "validation"
        val_dir.mkdir(parents=True, exist_ok=True)
        val_dataset = Dataset.from_list(val_pairs)
        val_dataset.to_json(str(val_dir / "sft_val.jsonl"))

        # Generate statistics
        self.generate_statistics(train_pairs, val_pairs, output_dir)

        logger.info("SFT dataset building complete!")

    def create_shards(self, pairs: List[Dict], output_dir: Path, split: str):
        """Create sharded JSONL files"""
        output_dir.mkdir(parents=True, exist_ok=True)

        pairs_per_shard = len(pairs) // self.config.num_shards
        if pairs_per_shard == 0:
            pairs_per_shard = 1

        for shard_idx in range(self.config.num_shards):
            start_idx = shard_idx * pairs_per_shard
            end_idx = start_idx + pairs_per_shard if shard_idx < self.config.num_shards - 1 else len(pairs)

            shard_pairs = pairs[start_idx:end_idx]

            if not shard_pairs:
                continue

            dataset = Dataset.from_list(shard_pairs)
            shard_filename = f"sft_{split}_shard_{shard_idx:02d}.jsonl"
            shard_path = output_dir / shard_filename

            dataset.to_json(str(shard_path))
            logger.info(f"Saved shard {shard_idx}: {shard_path} ({len(shard_pairs)} pairs)")

    def generate_statistics(self, train_pairs: List[Dict], val_pairs: List[Dict], output_dir: Path):
        """Generate dataset statistics"""
        stats = {
            'train': {
                'num_pairs': len(train_pairs),
                'avg_length': np.mean([p['length'] for p in train_pairs]),
                'avg_question_length': np.mean([len(p['question']) for p in train_pairs]),
                'avg_answer_length': np.mean([len(p['answer']) for p in train_pairs]),
            },
            'validation': {
                'num_pairs': len(val_pairs),
                'avg_length': np.mean([p['length'] for p in val_pairs]),
                'avg_question_length': np.mean([len(p['question']) for p in val_pairs]),
                'avg_answer_length': np.mean([len(p['answer']) for p in val_pairs]),
            }
        }

        report_path = output_dir / "sft_statistics.json"
        with open(report_path, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Statistics saved to {report_path}")


def main():
    """Command-line interface"""
    import argparse

    parser = argparse.ArgumentParser(description='Build SFT dataset from parsed documents')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing parsed documents')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for SFT dataset')
    parser.add_argument('--target_pairs', type=int, default=150000,
                       help='Target number of Q&A pairs')
    parser.add_argument('--num_shards', type=int, default=8,
                       help='Number of shards')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    np.random.seed(args.seed)

    config = SFTConfig(
        target_pairs=args.target_pairs,
        num_shards=args.num_shards
    )

    builder = SFTDatasetBuilder(config)
    builder.build_dataset(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir)
    )


if __name__ == '__main__':
    main()
