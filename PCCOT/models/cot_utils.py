import re
from typing import List, Sequence, Tuple, TypeVar


T = TypeVar("T")


def split_reference_and_superpose_steps(
    steps: Sequence[T],
    *,
    include_last_cot: bool,
    include_last_for_superpose_targets: bool,
) -> Tuple[List[T], List[T]]:
    all_steps = list(steps)
    reference_steps = list(all_steps if include_last_cot else all_steps[:-1])
    if include_last_cot or include_last_for_superpose_targets:
        superpose_steps = list(all_steps)
    else:
        superpose_steps = list(reference_steps)
    return reference_steps, superpose_steps


def prepare_superpose_step_text_for_tokenization(
    step_text,
    *,
    is_first_step: bool,
    strip_markers: bool = False,
    preserve_boundary_whitespace: bool = True,
) -> str:
    text = str(step_text) if step_text is not None else ""
    text = text.strip()
    if strip_markers and text.startswith("<<") and text.endswith(">>"):
        text = text[2:-2].strip()
    if not text:
        return ""
    if not preserve_boundary_whitespace:
        return text
    return text if is_first_step else f" {text}"
