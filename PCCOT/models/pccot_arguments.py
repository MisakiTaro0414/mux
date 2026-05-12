from dataclasses import dataclass, field
from pathlib import Path
import json


PCCOT_ARGS_NAME = "pccot_args.json"

@dataclass
class PCCoTArguments:
    """
    Arguments pertaining to the discourse transformer model.
    """
    # Tokenizer arguments
    bot_token_id: int = field(
        default=None,
        metadata={"help": "The pccot-special id of the beginning of sentence token. Please do not set this argument unless you understand the consequences."}
    )
    eot_token_id: int = field(
        default=None,
        metadata={"help": "The pccot-special id of the end of sentence token. Please do not set this argument unless you understand the consequences."}
    )
    latent_token_id: int = field(
        default=None,
        metadata={"help": "The pccot-special id of the sink token. Please do not set this argument unless you understand the consequences."}
    )
    label_pad_token_id: int = field(
        default=-100,
        metadata={"help": "The id of the padding token for labels. Labels with this id will not be taken into account in the loss computation. Please do not set this argument unless you understand the consequences."}
    )
    use_chat_template: bool = field(
        default=False,
        metadata={"help": "Whether to use the chat template for input formatting."}
    )

    # CoT specific arguments
    answer_prompt: str = field(
        default="The answer is:",
        metadata={"help": "The prompt to identify the answer in the input. The last token of the prompt will be used to do knowledge distillation."}
    )
    num_latent_tokens: int = field(
        default=6,
        metadata={"help": "The number of latent tokens to use in the PCCoT model."}
    )
    include_last_cot: bool = field(
        default=False,
        metadata={"help": "Whether to keep the final CoT step inside the teacher/reference path."}
    )
    ccot_kl_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature used when turning latent states into vocab-space KL targets."}
    )
    ccot_superpose_kl: bool = field(
        default=False,
        metadata={"help": "Enable CODI-style latent superpose KL supervision in PCCoT."}
    )
    ccot_superpose_kl_use_bag: bool = field(
        default=True,
        metadata={"help": "Use the bag-of-tokens superpose KL term."}
    )
    ccot_superpose_kl_scalar_positional: bool = field(
        default=False,
        metadata={"help": "Add scalar positional weighting to the superpose KL targets."}
    )
    ccot_superpose_kl_scalar_positional_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for the scalar-positional superpose KL term before beta scaling."}
    )
    ccot_superpose_kl_scalar_positional_mode: str = field(
        default="geometric",
        metadata={"help": "Scalar positional weighting mode for superpose KL."}
    )
    ccot_superpose_kl_scalar_positional_decay: float = field(
        default=0.9,
        metadata={"help": "Geometric decay used by scalar positional superpose KL."}
    )
    ccot_superpose_kl_scalar_positional_sin_amp: float = field(
        default=1.0,
        metadata={"help": "Amplitude used by sinusoidal-like scalar positional superpose KL."}
    )
    ccot_superpose_kl_scalar_positional_fourier_pairs: int = field(
        default=16,
        metadata={"help": "Number of Fourier pairs used by transformer/RoPE scalar positional modes."}
    )
    ccot_superpose_kl_scalar_positional_base: float = field(
        default=10000.0,
        metadata={"help": "Base used by transformer/RoPE scalar positional modes."}
    )
    ccot_superpose_kl_scalar_positional_freq_sampling: str = field(
        default="spread",
        metadata={"help": "Frequency sampling strategy for transformer/RoPE scalar positional modes."}
    )
    ccot_superpose_kl_scalar_positional_max_positions: int = field(
        default=128,
        metadata={"help": "Maximum table size for learned scalar positional weights."}
    )
    ccot_superpose_kl_teacher_distribution: bool = field(
        default=False,
        metadata={"help": "Use teacher-distribution superpose targets. Reserved for parity with CODI scripts."}
    )
    ccot_superpose_kl_teacher_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for teacher-distribution superpose targets."}
    )
    ccot_superpose_kl_train_lm_head: bool = field(
        default=False,
        metadata={"help": "Train a standalone LM head for superpose KL. Reserved for parity with CODI scripts."}
    )
    ccot_superpose_kl_train_lm_head_reinit: bool = field(
        default=False,
        metadata={"help": "Reinitialize the standalone superpose-KL LM head. Reserved for parity with CODI scripts."}
    )
    ccot_superpose_kl_target_q_feedback: bool = field(
        default=False,
        metadata={"help": "Enable q-feedback latent inputs. Reserved for parity with CODI scripts."}
    )
    ccot_superpose_kl_q_feedback_train_source: str = field(
        default="target",
        metadata={"help": "Training source for q-feedback. Reserved for parity with CODI scripts."}
    )
    ccot_superpose_uniform_chunk_to_latents: bool = field(
        default=False,
        metadata={"help": "Uniformly chunk CoT steps to latent supervision slots when there are more steps than slots."}
    )
    ccot_superpose_random_chunk_to_latents: bool = field(
        default=False,
        metadata={"help": "Randomly chunk CoT steps to latent supervision slots when there are more steps than slots."}
    )
    ccot_superpose_random_displace_steps_to_latents: bool = field(
        default=False,
        metadata={
            "help": (
                "Training-only augmentation: when there are fewer valid superpose targets than available "
                "supervision slots, randomly displace the ordered targets across the slots and mask the rest."
            )
        }
    )
    ccot_superpose_uniform_token_chunk_to_latents: bool = field(
        default=False,
        metadata={
            "help": (
                "Ignore original step boundaries and uniformly chunk the full tokenized superpose CoT chain "
                "into latent supervision slots. Extra tokens are assigned to earlier slots "
                "(e.g. 52 tokens over 25 slots -> [3, 3, 2, ..., 2])."
            )
        }
    )
    ccot_superpose_teacher_force: bool = field(
        default=False,
        metadata={"help": "Teacher-force latent slots from step targets. Reserved for parity with CODI scripts."}
    )
    ccot_superpose_teacher_force_poscoded: bool = field(
        default=False,
        metadata={"help": "Use positional-coded teacher forcing for latent slots. Reserved for parity with CODI scripts."}
    )
    ccot_superpose_beta: float = field(
        default=1.0,
        metadata={"help": "Weight applied to the total superpose KL auxiliary loss."}
    )
    ccot_superpose_beta_min: float = field(
        default=1.0,
        metadata={"help": "Reserved for CODI parity; PCCoT currently uses a constant superpose beta."}
    )
    ccot_superpose_strip_markers: bool = field(
        default=True,
        metadata={"help": "Strip << >> markers before tokenizing each superpose target step."}
    )
    ccot_superpose_include_last_cot_target_only: bool = field(
        default=True,
        metadata={"help": "Keep the final CoT step only in the superpose targets while leaving it out of the teacher path."}
    )
    ccot_superpose_k_plus_one_slots: bool = field(
        default=True,
        metadata={"help": "Supervise the prompt-end slot plus all latent slots instead of the latent slots only."}
    )
    ccot_superpose_num_supervised_latents: int = field(
        default=0,
        metadata={"help": "Reserved for CODI-script parity. PCCoT currently accepts only 0 for this option."}
    )

    # PEFT arguments
    use_peft: bool = field(
        default=True,
        metadata={"help": "Whether to use the PEFT strategy."}
    )
    lora_r: int = field(
        default=128,
        metadata={"help": "The rank of the LoRA matrix."}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "The weight to use for the LoRA loss."}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout to use in the LoRA matrix."}
    )
    lora_target_modules: str = field(
        default="q_proj-k_proj-v_proj-o_proj-down_proj-up_proj-gate_proj",
        metadata={"help": "hyphen separated list of target modules to apply LoRA layers to"},
    )
    lora_modules_to_save: str = field(
        default="",
        metadata={"help": "hyphen separated list of modules apart from adapter layers to be set as trainable and saved in the final checkpoint."},
    )

    def save(self, output_dir: Path):
        """ Save the arguments to a json file """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / PCCOT_ARGS_NAME, "w") as f:
            json.dump(vars(self), f, indent=4)
