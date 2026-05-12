from typing import List, Union
import torch
from torch.nn.utils.rnn import pad_sequence

from transformers import PreTrainedTokenizer
from transformers.utils import (
    is_sentencepiece_available,
    is_tokenizers_available,
)
from .pccot_arguments import PCCoTArguments
from .cot_utils import prepare_superpose_step_text_for_tokenization, split_reference_and_superpose_steps


def is_llama_tokenizer(tokenizer: PreTrainedTokenizer) -> bool:
    """Returns whether the tokenizer is a Llama tokenizer."""
    if is_sentencepiece_available():
        from transformers.models.llama import LlamaTokenizer
        if isinstance(tokenizer, LlamaTokenizer):
            return True
    
    if is_tokenizers_available():
        from transformers.models.llama import LlamaTokenizerFast
        if isinstance(tokenizer, LlamaTokenizerFast):
            return True

    return False


def batch_tokenize_number(tokenizer: PreTrainedTokenizer, number_texts: List[str]) -> list:
    """
    Tokenizes the number string into a list of tokens.

    This function is designed to handle the special case of the Llama tokenizer. The Llama tokenizer will
    add a space before the number token if the number is the first token in the sequence. This function
    will remove the space token if it is present.
    """
    input_ids = tokenizer(
        number_texts,
        return_attention_mask=False,
        add_special_tokens=False,
    )["input_ids"]

    if is_llama_tokenizer(tokenizer):
        # possibly extract the space token
        tokenized_1 = tokenizer.tokenize("123")
        tokenized_2 = tokenizer.tokenize("82435")
        if tokenized_1[0] == tokenized_2[0]:
            space_token_id = tokenizer.convert_tokens_to_ids(tokenized_1[0])
            input_ids = [tokens[1:] if tokens[0] == space_token_id else tokens for tokens in input_ids]

    return input_ids


class COTDataProcessor:
    """
    Data processor for the COT model.

    Example:
    {"question": "Janets ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?", "steps": ["<<16-3-4=9>>", "<<9*2=18>>"], "answer": "18"}
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        pccot_args: PCCoTArguments,
        max_seq_length: int = 8192,
    ) -> None:
        self.tokenizer = tokenizer
        self.pccot_args = pccot_args
        self.max_seq_length = max_seq_length

        self.tokenized_answer_prompt = self.tokenizer.encode(
            pccot_args.answer_prompt, add_special_tokens=False
        )
        self.superpose_enabled = bool(self.pccot_args.ccot_superpose_kl)

    def _tokenize_steps(self, steps, *, strip_markers: bool = False) -> list:
        tokenized_steps = []
        for step_idx, step in enumerate(steps):
            if self.superpose_enabled:
                step_text = prepare_superpose_step_text_for_tokenization(
                    step,
                    is_first_step=(step_idx == 0),
                    strip_markers=strip_markers,
                    preserve_boundary_whitespace=True,
                )
                if not step_text:
                    continue
                token_ids = self.tokenizer(
                    step_text,
                    return_attention_mask=False,
                    add_special_tokens=False,
                )["input_ids"]
            else:
                token_ids = self.tokenizer(
                    step,
                    return_attention_mask=False,
                    add_special_tokens=False,
                )["input_ids"]
            tokenized_steps.append(token_ids)
        return tokenized_steps

    def tokenize_function(self, examples):
        # if use chat template and the tokenizer has chat template, use it
        if (
            self.pccot_args.use_chat_template
            and hasattr(self.tokenizer, "chat_template")
            and self.tokenizer.chat_template
        ):
            question = [
                self.tokenizer.apply_chat_template([{"role": "user", "content": _example_question},], add_generation_prompt=True)
                for _example_question in examples["question"]
            ]
        else:
            question = self.tokenizer(
                examples["question"],
                return_attention_mask=False,
                add_special_tokens=True,
            )["input_ids"]

        reference_steps = []
        superpose_steps = []
        for steps in examples["steps"]:
            ref_steps, latent_steps = split_reference_and_superpose_steps(
                steps,
                include_last_cot=bool(self.pccot_args.include_last_cot),
                include_last_for_superpose_targets=bool(self.pccot_args.ccot_superpose_include_last_cot_target_only),
            )
            reference_steps.append(self._tokenize_steps(ref_steps, strip_markers=False))
            superpose_steps.append(
                self._tokenize_steps(
                    latent_steps,
                    strip_markers=bool(self.pccot_args.ccot_superpose_strip_markers),
                )
            )

        # tokenize the steps and answer
        output = {
            "question": question,
            "steps": reference_steps,
            "superpose_steps": superpose_steps,
            "answer": batch_tokenize_number(self.tokenizer, examples["answer"]),
        }
        return output

    def group_texts(self, examples):
        """
        We need to build cot sequence and ccot sequence.
        The cot sequence is built by concatenating the question, steps, and answer.
        The ccot sequence is built by concatenating the question, latent tokens, and answer.
        For both the sequences, we need to provide a list of indices to indicate the last token of the answer prompt.
        """
        # Building the cot sequence
        cot_input_ids = [
            question
            + [token for step in steps for token in step]
            + self.tokenized_answer_prompt
            + answer
            + [self.tokenizer.eos_token_id]
            for question, steps, answer in zip(
                examples["question"], examples["steps"], examples["answer"]
            )
        ]
        cot_labels = [
            [self.pccot_args.label_pad_token_id] * len(question)
            + [token for step in steps for token in step]
            + self.tokenized_answer_prompt
            + answer
            + [self.tokenizer.eos_token_id]
            for question, steps, answer in zip(
                examples["question"], examples["steps"], examples["answer"]
            )
        ]
        cot_kd_index = [
            len(question) + sum(len(step) for step in steps) + len(self.tokenized_answer_prompt) - 1
            for question, steps in zip(examples["question"], examples["steps"])
        ]
        # Building the ccot sequence without the question
        ccot_ids_woq = [
            [self.pccot_args.bot_token_id]
            + [self.pccot_args.latent_token_id] * self.pccot_args.num_latent_tokens
            + [self.pccot_args.eot_token_id]
            + self.tokenized_answer_prompt
            + answer
            + [self.tokenizer.eos_token_id]
            for answer in examples["answer"]
        ]
        ccot_kd_index = [
            2 + self.pccot_args.num_latent_tokens + len(self.tokenized_answer_prompt) - 1
            for _ in examples["answer"]
        ]
        return {
            "cot_input_ids": cot_input_ids,
            "cot_labels": cot_labels,
            "cot_kd_index": cot_kd_index,
            "ccot_ids_woq": ccot_ids_woq,
            "ccot_kd_index": ccot_kd_index,
            "superpose_steps": examples["superpose_steps"],
        }

    def data_collator(self, features):
        """
        We need to build cot sequence and ccot sequence, with proper padding.
        The cot sequence is built by concatenating the question, steps, and answer.
        The ccot sequence is built by concatenating the question, latent tokens, and answer. Using left padding for the questions.
        For both the sequences, we need to provide a list of indices to indicate the last token of the answer prompt.
        """
        # Padding the question, steps, and answer
        cot_input_ids = self.tokenizer.pad(
            {"input_ids": [item['cot_input_ids'] for item in features]},
            padding=True,
            return_tensors="pt",
        )
        cot_input_ids, attention_mask = cot_input_ids["input_ids"], cot_input_ids["attention_mask"]
        cot_labels = self.tokenizer.pad(
            {"input_ids": [item['cot_labels'] for item in features]},
            padding=True,
            return_tensors="pt",
        )["input_ids"]
        cot_labels[cot_labels == self.tokenizer.pad_token_id] = self.pccot_args.label_pad_token_id
        cot_kd_index = torch.tensor([item['cot_kd_index'] for item in features])

        questions = self.tokenizer.pad(
            {"input_ids": [item['question'] for item in features]},
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
        questions, question_attention_mask = questions["input_ids"], questions["attention_mask"]
        ccot_ids_woq = self.tokenizer.pad(
            {"input_ids": [item['ccot_ids_woq'] for item in features]},
            padding=True,
            return_tensors="pt",
        )
        ccot_ids_woq, ccot_woq_attention_mask = ccot_ids_woq["input_ids"], ccot_ids_woq["attention_mask"]
        ccot_input_ids = torch.cat([questions, ccot_ids_woq], dim=1)
        ccot_attention_mask = torch.cat([question_attention_mask, ccot_woq_attention_mask], dim=1)
        ccot_labels = ccot_ids_woq[:, 1 + self.pccot_args.num_latent_tokens:] # eot_token and the following tokens
        ccot_labels[ccot_labels == self.tokenizer.pad_token_id] = self.pccot_args.label_pad_token_id
        ccot_key_indices = [
            questions.size(1) + 1, # question + bot_token
            questions.size(1) + 1 + self.pccot_args.num_latent_tokens, # question + bot_token + latent_tokens
            1 + len(self.tokenized_answer_prompt) - 1 # eot_token + answer_prompt
        ]

        batch = {
            "input_ids": ccot_input_ids,
            "labels": ccot_labels,
            "attention_mask": ccot_attention_mask,
            "key_indices": ccot_key_indices,
            "cot_input_ids": cot_input_ids,
            "cot_labels": cot_labels,
            "cot_attention_mask": attention_mask,
            "cot_kd_indices": cot_kd_index,
        }
        if self.superpose_enabled:
            superpose_steps = [item["superpose_steps"] for item in features]
            max_steps = max((len(steps) for steps in superpose_steps), default=0)
            flat_entries = [
                (batch_idx, step_idx, step_ids)
                for batch_idx, steps in enumerate(superpose_steps)
                for step_idx, step_ids in enumerate(steps[:max_steps])
                if len(step_ids) > 0
            ]
            if flat_entries:
                flat_step_tensors = [
                    torch.tensor(step_ids, dtype=torch.long)
                    for _, _, step_ids in flat_entries
                ]
                padded_steps = pad_sequence(
                    flat_step_tensors,
                    batch_first=True,
                    padding_value=self.tokenizer.pad_token_id,
                )
                lengths = torch.tensor(
                    [step_tensor.size(0) for step_tensor in flat_step_tensors],
                    dtype=torch.long,
                )
                max_tokens = int(padded_steps.size(1))
                superpose_ids = torch.full(
                    (len(features), max_steps, max_tokens),
                    fill_value=self.tokenizer.pad_token_id,
                    dtype=torch.long,
                )
                superpose_token_mask = torch.zeros(
                    (len(features), max_steps, max_tokens),
                    dtype=torch.float,
                )
                superpose_step_mask = torch.zeros(
                    (len(features), max_steps),
                    dtype=torch.float,
                )

                batch_indices = torch.tensor([batch_idx for batch_idx, _, _ in flat_entries], dtype=torch.long)
                step_indices = torch.tensor([step_idx for _, step_idx, _ in flat_entries], dtype=torch.long)
                superpose_ids[batch_indices, step_indices] = padded_steps
                superpose_step_mask[batch_indices, step_indices] = 1.0

                token_positions = torch.arange(max_tokens, dtype=torch.long).unsqueeze(0)
                valid_token_mask = token_positions < lengths.unsqueeze(1)
                superpose_token_mask[batch_indices, step_indices] = valid_token_mask.to(dtype=torch.float)
            else:
                superpose_ids = torch.full(
                    (len(features), max_steps, 0),
                    fill_value=self.tokenizer.pad_token_id,
                    dtype=torch.long,
                )
                superpose_token_mask = torch.zeros(
                    (len(features), max_steps, 0),
                    dtype=torch.float,
                )
                superpose_step_mask = torch.zeros(
                    (len(features), max_steps),
                    dtype=torch.float,
                )
            batch["superpose_step_token_ids"] = superpose_ids
            batch["superpose_step_token_mask"] = superpose_token_mask
            batch["superpose_step_mask"] = superpose_step_mask

        return batch

    def process(self, questions: Union[str, List[str]], device: str = "cpu") -> dict:
        """
        Process the input questions and return the collated data without the answer and eos tokens.
        This function provides a unified interface for processing a single question or a list of questions,
        especially useful for inference scenarios where you do not have a dataset.
        
        Args:
            questions (Union[str, List[str]]): A single question or a list of questions.
            device (str): The device to which the processed data should be moved (default is "cpu").
        
        Returns:
            dict: A dictionary containing the processed data.
        """
        if isinstance(questions, str):
            questions = [questions]
        
        num_examples = len(questions)
        
        tokenized = self.tokenize_function({
            "question": questions,
            "steps": [[]] * num_examples,
            "answer": [""] * num_examples,
        })
        
        grouped = self.group_texts(tokenized)
        grouped = {**tokenized, **grouped}
        grouped = [
            {
                k: v[i]
                for k, v in grouped.items()
            }
            for i in range(len(grouped["question"]))
        ]
        
        collated = self.data_collator(grouped)

        # remove eos
        collated["input_ids"] = collated["input_ids"][:, :-1]
        collated["labels"] = collated["labels"][:, :-1]
        collated["attention_mask"] = collated["attention_mask"][:, :-1]

        # Move to the specified device
        for k, v in collated.items():
            if isinstance(v, torch.Tensor):
                collated[k] = v.to(device)

        return collated
