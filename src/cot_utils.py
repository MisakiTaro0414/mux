import re
from typing import List, Sequence, Tuple, TypeVar


T = TypeVar("T")


def split_reference_and_superpose_steps(
    steps: Sequence[T],
    *,
    include_last_cot: bool,
    include_last_for_superpose_targets: bool,
) -> Tuple[List[T], List[T]]:
    """Return separate step sequences for the reference path and superpose targets."""
    all_steps = list(steps)
    reference_steps = list(all_steps if include_last_cot else all_steps[:-1])
    if include_last_cot or include_last_for_superpose_targets:
        superpose_steps = list(all_steps)
    else:
        superpose_steps = list(reference_steps)
    return reference_steps, superpose_steps


def split_natural_cot_sentences(cot_text: str) -> List[str]:
    """Split natural-language CoT text into sentence steps while keeping trailing periods."""
    cot_text = str(cot_text).strip() if cot_text is not None else ""
    if not cot_text:
        return []
    return [part.strip() for part in re.split(r"(?<=\.)\s+", cot_text) if part and part.strip()]


def extract_superpose_steps(
    cot_target,
    *,
    natural_language_steps: bool = False,
) -> List[str]:
    """
    Extract superpose targets from either:
    - a pre-split sequence of steps
    - structured <<...>> markup
    - raw text fallback, optionally sentence-split for natural-language CoTs
    """
    if cot_target is None:
        return []
    if isinstance(cot_target, (list, tuple)):
        return [str(step).strip() for step in cot_target if str(step).strip()]

    cot_text = str(cot_target).strip()
    if not cot_text:
        return []

    marked = re.findall(r"<<[^<>]*>>", cot_text)
    if marked:
        return marked
    if natural_language_steps:
        return split_natural_cot_sentences(cot_text)
    return [step for step in cot_text.split(" ") if step]


def prepare_superpose_step_text_for_tokenization(
    step_text,
    *,
    is_first_step: bool,
    strip_markers: bool = False,
    preserve_boundary_whitespace: bool = True,
) -> str:
    """
    Normalize one superpose step for standalone tokenization while preserving the
    boundary whitespace it has inside the reference CoT string.

    This matters for tokenizers where `"foo"` and `" foo"` produce different ids.
    """
    text = str(step_text) if step_text is not None else ""
    text = text.strip()
    if strip_markers and text.startswith("<<") and text.endswith(">>"):
        text = text[2:-2].strip()
    if not text:
        return ""
    if not preserve_boundary_whitespace:
        return text
    return text if is_first_step else f" {text}"


def find_contiguous_subsequence_offset(
    sequence: Sequence[int],
    subsequence: Sequence[int],
) -> int:
    """Return the starting offset of a contiguous subsequence inside a sequence."""
    seq = list(sequence)
    sub = list(subsequence)
    if not sub:
        return 0
    sub_len = len(sub)
    max_start = len(seq) - sub_len
    for start in range(max_start + 1):
        if seq[start : start + sub_len] == sub:
            return start
    raise ValueError("Subsequence not found in sequence.")
