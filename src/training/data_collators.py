"""
Data Collators for Training

Includes specialized collators for:
- CPT: Standard language modeling collator
- SFT: Input masking collator (only compute loss on assistant responses)
"""

import torch
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from transformers import PreTrainedTokenizer


@dataclass
class DataCollatorForCPT:
    """
    Data collator for Continued Pretraining (CPT)

    Standard language modeling collator with padding
    """
    tokenizer: PreTrainedTokenizer
    max_length: int = 4096

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate features for CPT training

        Args:
            features: List of dictionaries with 'input_ids'

        Returns:
            Batch dictionary with input_ids, attention_mask, and labels
        """
        # Extract input_ids
        input_ids = [f['input_ids'] for f in features]

        # Pad sequences
        batch = self.tokenizer.pad(
            {'input_ids': input_ids},
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Labels are same as input_ids for language modeling
        batch['labels'] = batch['input_ids'].clone()

        # Mask padding tokens in labels (-100 is ignored in loss)
        batch['labels'][batch['labels'] == self.tokenizer.pad_token_id] = -100

        return batch


@dataclass
class DataCollatorForSFT:
    """
    Data collator for Supervised Fine-Tuning (SFT) with input masking

    Only computes loss on assistant's response, not on system prompt or user question.
    This prevents the model from learning to repeat instructions.
    """
    tokenizer: PreTrainedTokenizer
    max_length: int = 2048

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate features for SFT training with input masking

        Args:
            features: List of dictionaries with 'text' (formatted chat)

        Returns:
            Batch dictionary with input_ids, attention_mask, and labels (masked)
        """
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []

        for feature in features:
            # Get input text
            text = feature.get('text', '')

            # Tokenize
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()

            # Create labels with input masking
            labels = input_ids.clone()

            # Find where assistant response starts
            # LLaMA 3.1 chat format: <|start_header_id|>assistant<|end_header_id|>\n\n{response}
            assistant_start_idx = self._find_assistant_start(input_ids)

            # Mask everything before assistant response
            if assistant_start_idx > 0:
                labels[:assistant_start_idx] = -100

            # Mask padding tokens
            labels[labels == self.tokenizer.pad_token_id] = -100

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_attention_mask.append(attention_mask)

        return {
            'input_ids': torch.stack(batch_input_ids),
            'attention_mask': torch.stack(batch_attention_mask),
            'labels': torch.stack(batch_labels)
        }

    def _find_assistant_start(self, input_ids: torch.Tensor) -> int:
        """
        Find the index where assistant response starts

        LLaMA 3.1 format:
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>...
        <|start_header_id|>user<|end_header_id|>...
        <|start_header_id|>assistant<|end_header_id|>
        {RESPONSE}

        We want to start computing loss after <|end_header_id|> following "assistant"
        """
        # Find special tokens
        header_id_token = self.tokenizer.convert_tokens_to_ids('<|start_header_id|>')
        end_header_token = self.tokenizer.convert_tokens_to_ids('<|end_header_id|>')

        # Convert to list for easier searching
        ids_list = input_ids.tolist()

        # Find "assistant" token
        assistant_token_str = 'assistant'
        assistant_tokens = self.tokenizer.encode(assistant_token_str, add_special_tokens=False)

        # Search for the pattern: <|start_header_id|> assistant <|end_header_id|>
        for i in range(len(ids_list) - len(assistant_tokens) - 2):
            if ids_list[i] == header_id_token:
                # Check if next tokens are "assistant"
                match = True
                for j, token in enumerate(assistant_tokens):
                    if i + 1 + j >= len(ids_list) or ids_list[i + 1 + j] != token:
                        match = False
                        break

                if match:
                    # Found "assistant" header, find the closing <|end_header_id|>
                    search_start = i + 1 + len(assistant_tokens)
                    for k in range(search_start, min(search_start + 10, len(ids_list))):
                        if ids_list[k] == end_header_token:
                            # Start computing loss after <|end_header_id|> + newline tokens
                            return k + 1

        # Fallback: if pattern not found, mask first 50% of sequence
        return len(ids_list) // 2


@dataclass
class DataCollatorForSFTWithMetadata:
    """
    Extended SFT collator that preserves metadata for evaluation
    """
    tokenizer: PreTrainedTokenizer
    max_length: int = 2048

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate with metadata preservation"""
        base_collator = DataCollatorForSFT(self.tokenizer, self.max_length)
        batch = base_collator(features)

        # Add metadata
        batch['metadata'] = [f.get('metadata', {}) for f in features]
        batch['questions'] = [f.get('question', '') for f in features]
        batch['answers'] = [f.get('answer', '') for f in features]

        return batch


def test_input_masking():
    """Test input masking collator"""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")

    # Create sample data
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False)

    feature = {'text': text}

    collator = DataCollatorForSFT(tokenizer, max_length=512)
    batch = collator([feature])

    print("Input IDs shape:", batch['input_ids'].shape)
    print("Labels shape:", batch['labels'].shape)

    # Count masked tokens
    masked_count = (batch['labels'] == -100).sum().item()
    total_count = batch['labels'].numel()

    print(f"Masked tokens: {masked_count}/{total_count} ({masked_count/total_count*100:.1f}%)")

    # Decode to verify
    print("\nFull text:")
    print(tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False))

    print("\nLabels (response only):")
    labels_no_mask = batch['labels'][0].clone()
    labels_no_mask[labels_no_mask == -100] = tokenizer.pad_token_id
    print(tokenizer.decode(labels_no_mask, skip_special_tokens=True))


if __name__ == '__main__':
    print("Testing input masking collator...")
    test_input_masking()
