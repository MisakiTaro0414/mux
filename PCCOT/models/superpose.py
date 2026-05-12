import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class PCCoTSuperposeMixin:
    def _init_superpose_state(self) -> None:
        cfg = self.config
        self.superpose_kl_scalar_positional_logits = None
        if not bool(getattr(cfg, "ccot_superpose_kl", False)):
            return

        if bool(getattr(cfg, "ccot_superpose_kl_teacher_distribution", False)):
            raise NotImplementedError("Teacher-distribution superpose KL is not implemented in PCCoT yet.")
        if bool(getattr(cfg, "ccot_superpose_kl_target_q_feedback", False)):
            raise NotImplementedError("q-feedback superpose KL is not implemented in PCCoT yet.")
        if bool(getattr(cfg, "ccot_superpose_teacher_force", False)):
            raise NotImplementedError("Teacher-forced superpose latents are not implemented in PCCoT yet.")
        if bool(getattr(cfg, "ccot_superpose_kl_train_lm_head", False)) or bool(
            getattr(cfg, "ccot_superpose_kl_train_lm_head_reinit", False)
        ):
            raise NotImplementedError("Trainable standalone superpose-KL LM heads are not implemented in PCCoT yet.")
        use_step_uniform_chunk = bool(getattr(cfg, "ccot_superpose_uniform_chunk_to_latents", False))
        use_step_random_chunk = bool(getattr(cfg, "ccot_superpose_random_chunk_to_latents", False))
        use_token_uniform_chunk = bool(getattr(cfg, "ccot_superpose_uniform_token_chunk_to_latents", False))
        if use_step_uniform_chunk and use_step_random_chunk:
            raise ValueError(
                "Enable only one of ccot_superpose_uniform_chunk_to_latents or ccot_superpose_random_chunk_to_latents."
            )
        if use_token_uniform_chunk and (use_step_uniform_chunk or use_step_random_chunk):
            raise ValueError(
                "ccot_superpose_uniform_token_chunk_to_latents is incompatible with step-based chunking flags. "
                "Enable only one chunking strategy."
            )
        if float(getattr(cfg, "ccot_kl_temperature", 1.0)) <= 0.0:
            raise ValueError("ccot_kl_temperature must be > 0.")
        if float(getattr(cfg, "ccot_superpose_beta", 1.0)) < 0.0:
            raise ValueError("ccot_superpose_beta must be >= 0.")
        if int(getattr(cfg, "ccot_superpose_num_supervised_latents", 0)) != 0:
            raise NotImplementedError(
                "ccot_superpose_num_supervised_latents>0 is not implemented in PCCoT yet."
            )

        if self._resolve_scalar_positional_mode() == "learned":
            max_pos = max(1, int(getattr(cfg, "ccot_superpose_kl_scalar_positional_max_positions", 128)))
            self.superpose_kl_scalar_positional_logits = nn.Parameter(torch.zeros(max_pos))

    def _resolve_scalar_positional_mode(self, mode: Optional[str] = None) -> str:
        resolved = str(
            getattr(self.config, "ccot_superpose_kl_scalar_positional_mode", "geometric") if mode is None else mode
        ).strip().lower()
        if resolved in {"geo", "exp"}:
            resolved = "geometric"
        elif resolved in {"sin", "sine"}:
            resolved = "sinusoidal"
        elif resolved in {"transformer", "transformer_sin", "tf_sinusoidal", "vaswani"}:
            resolved = "transformer_sinusoidal"
        elif resolved in {"rope", "rope_like", "rope-scalar"}:
            resolved = "rope"
        elif resolved in {"learn", "table"}:
            resolved = "learned"
        elif resolved in {"fb", "forward_backward", "bidirectional"}:
            resolved = "fb"
        if resolved not in {"geometric", "sinusoidal", "transformer_sinusoidal", "rope", "learned", "fb"}:
            raise ValueError(
                f"Unsupported ccot_superpose_kl_scalar_positional_mode={resolved!r}. "
                "Use one of: geometric, sinusoidal, transformer_sinusoidal, rope, learned, fb."
            )
        return resolved

    def _entropy_from_token_weights(
        self,
        token_ids: torch.LongTensor,
        token_weights: torch.Tensor,
    ) -> torch.Tensor:
        if token_ids.ndim != 3 or token_weights.ndim != 3:
            raise ValueError(
                "Expected token_ids/token_weights to be [B,S,T], "
                f"got {tuple(token_ids.shape)} and {tuple(token_weights.shape)}"
            )
        if token_ids.shape != token_weights.shape:
            raise ValueError(
                f"Expected token_ids/token_weights to have identical shapes, got "
                f"{tuple(token_ids.shape)} and {tuple(token_weights.shape)}"
            )

        bsz, num_steps, t_len = token_ids.shape
        if num_steps == 0 or t_len == 0:
            return torch.zeros((bsz, num_steps), device=token_weights.device, dtype=token_weights.dtype)

        flat_ids = token_ids.reshape(-1, t_len)
        flat_weights = token_weights.reshape(-1, t_len)
        safe_ids = torch.where(flat_weights > 0, flat_ids, torch.zeros_like(flat_ids))
        sorted_ids, perm = torch.sort(safe_ids, dim=-1)
        sorted_weights = flat_weights.gather(dim=-1, index=perm)

        group_idx = torch.zeros_like(sorted_ids, dtype=torch.long)
        if t_len > 1:
            group_idx[:, 1:] = (sorted_ids[:, 1:] != sorted_ids[:, :-1]).to(dtype=torch.long)
            group_idx = group_idx.cumsum(dim=-1)

        group_mass = torch.zeros_like(sorted_weights)
        group_mass.scatter_add_(dim=-1, index=group_idx, src=sorted_weights)
        entropy = -(group_mass * torch.log(group_mass.clamp(min=1e-12))).sum(dim=-1)
        return entropy.view(bsz, num_steps)

    def _chunk_superpose_steps_to_latents(
        self,
        target_token_ids: torch.LongTensor,
        target_token_mask: torch.Tensor,
        step_mask: Optional[torch.Tensor] = None,
        num_latent: Optional[int] = None,
        randomize: bool = False,
        chunk_sizes: Optional[list[list[int]]] = None,
    ) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor]:
        if target_token_ids is None or target_token_mask is None:
            raise RuntimeError("Superpose chunking requires target_token_ids and target_token_mask.")
        if target_token_ids.ndim != 3 or target_token_mask.ndim != 3:
            raise ValueError(
                "Expected target_token_ids/target_token_mask to be [B,S,T], "
                f"got {tuple(target_token_ids.shape)} and {tuple(target_token_mask.shape)}"
            )

        k = int(target_token_ids.size(1) if num_latent is None else num_latent)
        if k < 0:
            raise ValueError(f"num_latent must be non-negative, got {k}")

        device = target_token_ids.device
        ids = target_token_ids.to(device)
        tok_mask_in = target_token_mask.to(device)
        tok_mask_bool = tok_mask_in.bool()
        bsz, _, _ = ids.shape

        mask_dtype = (
            tok_mask_in.dtype
            if (tok_mask_in.dtype.is_floating_point or tok_mask_in.dtype == torch.bool)
            else torch.float32
        )
        token_counts = tok_mask_bool.sum(dim=-1)
        valid_steps = token_counts > 0
        if step_mask is not None:
            valid_steps = valid_steps & step_mask.to(device).bool()

        if k == 0:
            chunk_ids = torch.zeros((bsz, 0, 0), device=device, dtype=ids.dtype)
            chunk_tok_mask = torch.zeros((bsz, 0, 0), device=device, dtype=mask_dtype)
            chunk_step_mask = torch.zeros((bsz, 0), device=device, dtype=mask_dtype)
            return chunk_ids, chunk_tok_mask, chunk_step_mask
        num_valid = valid_steps.sum(dim=1)
        if chunk_sizes is not None:
            sizes = torch.as_tensor(chunk_sizes, device=device, dtype=torch.long)
            if sizes.shape != (bsz, k):
                raise ValueError(f"Expected chunk_sizes to have shape {(bsz, k)}, got {tuple(sizes.shape)}")
            if torch.any(sizes < 0):
                raise ValueError("Chunk plan sizes must be non-negative.")
            if not torch.equal(sizes.sum(dim=1), num_valid):
                raise ValueError("Chunk plan must cover all valid steps.")
        else:
            sizes = torch.zeros((bsz, k), device=device, dtype=torch.long)
            slot_idx = torch.arange(k, device=device, dtype=torch.long).view(1, k)

            needs_prefix = num_valid <= k
            if bool(needs_prefix.any()):
                sizes[needs_prefix] = (slot_idx < num_valid[needs_prefix].unsqueeze(1)).to(dtype=torch.long)

            needs_multi = num_valid > k
            if bool(needs_multi.any()):
                if randomize:
                    if k == 1:
                        sizes[needs_multi, 0] = num_valid[needs_multi]
                    else:
                        num_rows = int(needs_multi.sum().item())
                        max_steps = ids.size(1)
                        boundary_scores = torch.rand((num_rows, max_steps - 1), device=device)
                        valid_boundaries = torch.arange(max_steps - 1, device=device).view(1, -1) < (
                            num_valid[needs_multi] - 1
                        ).unsqueeze(1)
                        boundary_scores = boundary_scores.masked_fill(~valid_boundaries, -1.0)
                        cut_points = boundary_scores.topk(k - 1, dim=1).indices.add(1)
                        cut_points, _ = torch.sort(cut_points, dim=1)
                        boundaries = torch.cat(
                            [
                                torch.zeros((num_rows, 1), device=device, dtype=torch.long),
                                cut_points,
                                num_valid[needs_multi].unsqueeze(1),
                            ],
                            dim=1,
                        )
                        sizes[needs_multi] = boundaries[:, 1:] - boundaries[:, :-1]
                else:
                    base = torch.div(num_valid[needs_multi], k, rounding_mode="floor")
                    rem = num_valid[needs_multi] % k
                    sizes_multi = base.unsqueeze(1).expand(-1, k).clone()
                    sizes_multi = sizes_multi + (
                        slot_idx.expand(sizes_multi.size(0), -1) >= (k - rem).unsqueeze(1)
                    ).to(dtype=torch.long)
                    sizes[needs_multi] = sizes_multi

        if not torch.equal(sizes.sum(dim=1), num_valid):
            raise RuntimeError("Resolved chunk sizes do not cover all valid steps.")

        step_ranks = valid_steps.to(dtype=torch.long).cumsum(dim=1) - 1
        slot_starts = sizes.cumsum(dim=1) - sizes
        slot_ends = slot_starts + sizes
        slot_assign = (
            valid_steps.unsqueeze(-1)
            & (step_ranks.unsqueeze(-1) >= slot_starts.unsqueeze(1))
            & (step_ranks.unsqueeze(-1) < slot_ends.unsqueeze(1))
        )
        assigned_steps = slot_assign.sum(dim=-1)
        if not torch.equal(assigned_steps, valid_steps.to(dtype=assigned_steps.dtype)):
            raise RuntimeError("Each valid superpose step must map to exactly one latent slot.")

        step_slot_idx = slot_assign.to(dtype=torch.long).argmax(dim=-1)
        step_token_counts = token_counts.unsqueeze(-1) * slot_assign.to(dtype=token_counts.dtype)
        slot_token_counts = step_token_counts.sum(dim=1)
        padded_len = max(1, int(slot_token_counts.max().item()))

        token_offsets = step_token_counts.cumsum(dim=1) - step_token_counts
        step_offsets = token_offsets.gather(dim=-1, index=step_slot_idx.unsqueeze(-1)).squeeze(-1)
        step_offsets = torch.where(valid_steps, step_offsets, torch.zeros_like(step_offsets))

        chunk_ids = torch.zeros((bsz, k, padded_len), device=device, dtype=ids.dtype)
        chunk_tok_mask = torch.zeros((bsz, k, padded_len), device=device, dtype=mask_dtype)
        chunk_step_mask = (slot_token_counts > 0).to(device=device, dtype=mask_dtype)

        valid_tokens = tok_mask_bool & valid_steps.unsqueeze(-1)
        if bool(valid_tokens.any()):
            token_pos = torch.arange(ids.size(-1), device=device, dtype=torch.long).view(1, 1, -1)
            batch_index = torch.arange(bsz, device=device, dtype=torch.long).view(bsz, 1, 1).expand_as(ids)
            token_slot_idx = step_slot_idx.unsqueeze(-1).expand_as(ids)
            chunk_positions = step_offsets.unsqueeze(-1) + token_pos
            chunk_ids[
                batch_index[valid_tokens],
                token_slot_idx[valid_tokens],
                chunk_positions[valid_tokens],
            ] = ids[valid_tokens]
            if chunk_tok_mask.dtype == torch.bool:
                chunk_tok_mask[
                    batch_index[valid_tokens],
                    token_slot_idx[valid_tokens],
                    chunk_positions[valid_tokens],
                ] = True
            else:
                chunk_tok_mask[
                    batch_index[valid_tokens],
                    token_slot_idx[valid_tokens],
                    chunk_positions[valid_tokens],
                ] = 1
        return chunk_ids, chunk_tok_mask, chunk_step_mask

    def _chunk_superpose_tokens_to_latents(
        self,
        target_token_ids: torch.LongTensor,
        target_token_mask: torch.Tensor,
        step_mask: Optional[torch.Tensor] = None,
        num_latent: Optional[int] = None,
    ) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor]:
        """
        Ignore step boundaries and uniformly split the full valid token chain into latent-aligned chunks.

        If a sample has N valid superpose tokens and K latent slots, chunk sizes are:
          floor(N / K) for all slots, plus one extra token for the first N % K slots.

        This is intended for datasets where there are only a few coarse steps but each step
        contains many tokens, so latent supervision should align to token-level local spans
        instead of the original step segmentation.
        """
        if target_token_ids is None or target_token_mask is None:
            raise RuntimeError("Superpose token chunking requires target_token_ids and target_token_mask.")
        if target_token_ids.ndim != 3 or target_token_mask.ndim != 3:
            raise ValueError(
                "Expected target_token_ids/target_token_mask to be [B,S,T], "
                f"got {tuple(target_token_ids.shape)} and {tuple(target_token_mask.shape)}"
            )

        k = int(target_token_ids.size(1) if num_latent is None else num_latent)
        if k < 0:
            raise ValueError(f"num_latent must be non-negative, got {k}")

        device = target_token_ids.device
        ids = target_token_ids.to(device)
        tok_mask_in = target_token_mask.to(device)
        tok_mask_bool = tok_mask_in.bool()
        bsz, _, _ = ids.shape

        mask_dtype = (
            tok_mask_in.dtype
            if (tok_mask_in.dtype.is_floating_point or tok_mask_in.dtype == torch.bool)
            else torch.float32
        )

        if k == 0:
            chunk_ids = torch.zeros((bsz, 0, 0), device=device, dtype=ids.dtype)
            chunk_tok_mask = torch.zeros((bsz, 0, 0), device=device, dtype=mask_dtype)
            chunk_step_mask = torch.zeros((bsz, 0), device=device, dtype=mask_dtype)
            return chunk_ids, chunk_tok_mask, chunk_step_mask

        token_counts = tok_mask_bool.sum(dim=-1)
        valid_steps = token_counts > 0
        if step_mask is not None:
            valid_steps = valid_steps & step_mask.to(device).bool()
        valid_tokens = tok_mask_bool & valid_steps.unsqueeze(-1)
        total_tokens = valid_tokens.sum(dim=(1, 2))

        slot_idx = torch.arange(k, device=device, dtype=torch.long).view(1, k)
        base = torch.div(total_tokens, k, rounding_mode="floor")
        remainder = total_tokens % k
        sizes = base.unsqueeze(1).expand(-1, k).clone()
        sizes = sizes + (slot_idx < remainder.unsqueeze(1)).to(dtype=torch.long)
        if not torch.equal(sizes.sum(dim=1), total_tokens):
            raise RuntimeError("Resolved token chunk sizes do not cover all valid tokens.")

        flat_ids = ids.reshape(bsz, -1)
        flat_valid = valid_tokens.reshape(bsz, -1)
        packed_len = max(1, int(total_tokens.max().item()))
        packed_ids = torch.zeros((bsz, packed_len), device=device, dtype=ids.dtype)
        if bool(flat_valid.any()):
            flat_ranks = flat_valid.to(dtype=torch.long).cumsum(dim=1) - 1
            batch_index = torch.arange(bsz, device=device, dtype=torch.long).view(bsz, 1).expand_as(flat_ids)
            packed_ids[batch_index[flat_valid], flat_ranks[flat_valid]] = flat_ids[flat_valid]

        padded_len = max(1, int(sizes.max().item()))
        slot_starts = sizes.cumsum(dim=1) - sizes
        local_pos = torch.arange(padded_len, device=device, dtype=torch.long).view(1, 1, padded_len)
        chunk_valid = local_pos < sizes.unsqueeze(-1)
        gather_idx = slot_starts.unsqueeze(-1) + local_pos
        gather_idx = gather_idx.clamp(max=packed_len - 1)

        chunk_ids = packed_ids.unsqueeze(1).expand(-1, k, -1).gather(dim=2, index=gather_idx)
        chunk_ids = chunk_ids.masked_fill(~chunk_valid, 0)
        chunk_tok_mask = chunk_valid.to(device=device, dtype=mask_dtype)
        chunk_step_mask = (sizes > 0).to(device=device, dtype=mask_dtype)
        return chunk_ids, chunk_tok_mask, chunk_step_mask

    def _randomly_displace_superpose_steps_to_latents(
        self,
        target_token_ids: torch.LongTensor,
        target_token_mask: torch.Tensor,
        step_mask: Optional[torch.Tensor] = None,
        num_latent: Optional[int] = None,
    ) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor]:
        """
        Randomly place ordered superpose targets into a subset of supervision slots.

        This is a training-time augmentation for the underflow regime (fewer valid targets than
        available slots). For each sample with M valid targets and K slots:
          - If M >= K, keep the default prefix alignment for the first K targets.
          - If M < K, sample M distinct slot positions uniformly without replacement from K slots,
            sort them to preserve target order, and mask the remaining slots.
        """
        if target_token_ids is None or target_token_mask is None:
            raise RuntimeError("Random superpose displacement requires target_token_ids and target_token_mask.")
        if target_token_ids.ndim != 3 or target_token_mask.ndim != 3:
            raise ValueError(
                "Expected target_token_ids/target_token_mask to be [B,S,T], "
                f"got {tuple(target_token_ids.shape)} and {tuple(target_token_mask.shape)}"
            )

        k = int(target_token_ids.size(1) if num_latent is None else num_latent)
        if k < 0:
            raise ValueError(f"num_latent must be non-negative, got {k}")

        device = target_token_ids.device
        ids = target_token_ids.to(device)
        tok_mask_in = target_token_mask.to(device)
        tok_mask_bool = tok_mask_in.bool()
        bsz, _, step_token_len = ids.shape

        mask_dtype = (
            tok_mask_in.dtype
            if (tok_mask_in.dtype.is_floating_point or tok_mask_in.dtype == torch.bool)
            else torch.float32
        )

        if k == 0:
            displaced_ids = torch.zeros((bsz, 0, step_token_len), device=device, dtype=ids.dtype)
            displaced_tok_mask = torch.zeros((bsz, 0, step_token_len), device=device, dtype=mask_dtype)
            displaced_step_mask = torch.zeros((bsz, 0), device=device, dtype=mask_dtype)
            return displaced_ids, displaced_tok_mask, displaced_step_mask

        token_counts = tok_mask_bool.sum(dim=-1)
        valid_steps = token_counts > 0
        if step_mask is not None:
            valid_steps = valid_steps & step_mask.to(device).bool()

        step_ranks = valid_steps.to(dtype=torch.long).cumsum(dim=1) - 1
        assigned_steps = valid_steps & (step_ranks < k)
        num_assigned = assigned_steps.sum(dim=1)
        apply_random = num_assigned < k

        prefix_slots = step_ranks.clamp(min=0, max=max(k - 1, 0))

        random_perm = torch.rand((bsz, k), device=device).argsort(dim=1)
        slot_rank = torch.arange(k, device=device, dtype=torch.long).view(1, k)
        selected_mask = slot_rank < num_assigned.unsqueeze(1)
        selected_slots = torch.where(selected_mask, random_perm, torch.full_like(random_perm, k))
        selected_slots, _ = torch.sort(selected_slots, dim=1)
        selected_slots = selected_slots.clamp(max=max(k - 1, 0))
        gather_idx = step_ranks.clamp(min=0, max=max(k - 1, 0))
        random_slots = selected_slots.gather(dim=1, index=gather_idx)

        step_slot_idx = torch.where(apply_random.unsqueeze(1), random_slots, prefix_slots)

        displaced_ids = torch.zeros((bsz, k, step_token_len), device=device, dtype=ids.dtype)
        displaced_tok_mask = torch.zeros((bsz, k, step_token_len), device=device, dtype=mask_dtype)
        displaced_step_mask = torch.zeros((bsz, k), device=device, dtype=mask_dtype)

        if bool(assigned_steps.any()):
            batch_index = torch.arange(bsz, device=device, dtype=torch.long).view(bsz, 1).expand_as(step_ranks)
            displaced_ids[
                batch_index[assigned_steps],
                step_slot_idx[assigned_steps],
            ] = ids[assigned_steps]
            displaced_tok_mask[
                batch_index[assigned_steps],
                step_slot_idx[assigned_steps],
            ] = tok_mask_in[assigned_steps].to(dtype=mask_dtype)
            if displaced_step_mask.dtype == torch.bool:
                displaced_step_mask[
                    batch_index[assigned_steps],
                    step_slot_idx[assigned_steps],
                ] = True
            else:
                displaced_step_mask[
                    batch_index[assigned_steps],
                    step_slot_idx[assigned_steps],
                ] = 1

        return displaced_ids, displaced_tok_mask, displaced_step_mask

    def _kl_from_superposed_token_targets(
        self,
        student_logits: torch.Tensor,
        target_token_ids: torch.LongTensor,
        target_token_mask: torch.Tensor,
        step_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        if student_logits is None:
            raise RuntimeError("student_logits is required for superposed-token KL.")
        if target_token_ids is None or target_token_mask is None:
            raise RuntimeError("Superposed-token KL requires target step token ids and masks in the batch.")
        if student_logits.ndim != 3:
            raise ValueError(f"Expected student_logits to be [B,K,V], got shape={tuple(student_logits.shape)}")
        if target_token_ids.ndim != 3 or target_token_mask.ndim != 3:
            raise ValueError(
                "Expected target_token_ids/target_token_mask to be [B,S,T], "
                f"got {tuple(target_token_ids.shape)} and {tuple(target_token_mask.shape)}"
            )

        device = student_logits.device
        _, num_slots, _ = student_logits.shape
        _, num_steps, _ = target_token_ids.shape
        aligned_steps = min(num_slots, num_steps)
        if aligned_steps <= 0:
            return torch.tensor(0.0, device=device, dtype=student_logits.dtype)

        logits = student_logits[:, :aligned_steps, :]
        if temperature != 1.0:
            logits = logits / float(temperature)
        log_probs = F.log_softmax(logits, dim=-1)

        ids = target_token_ids[:, :aligned_steps, :].to(device)
        token_mask = target_token_mask[:, :aligned_steps, :].to(device=device, dtype=log_probs.dtype)
        token_counts = token_mask.sum(dim=-1)

        if step_mask is not None:
            valid_step_mask = step_mask[:, :aligned_steps].to(device=device, dtype=log_probs.dtype)
            valid_step_mask = valid_step_mask * (token_counts > 0).to(log_probs.dtype)
        else:
            valid_step_mask = (token_counts > 0).to(log_probs.dtype)
        if float(valid_step_mask.sum().detach()) <= 0.0:
            return torch.tensor(0.0, device=device, dtype=student_logits.dtype)

        safe_ids = torch.where(token_mask > 0, ids, torch.zeros_like(ids))
        gathered_log_probs = log_probs.gather(dim=-1, index=safe_ids)
        ce_per_step = -((gathered_log_probs * token_mask).sum(dim=-1) / token_counts.clamp(min=1.0))

        with torch.no_grad():
            entropy_per_step = self._entropy_from_token_weights(
                token_ids=ids,
                token_weights=token_mask / token_counts.clamp(min=1.0).unsqueeze(-1),
            )

        kl_per_step = (ce_per_step - entropy_per_step) * valid_step_mask
        denom = valid_step_mask.sum(dim=1).clamp(min=1.0)
        return (kl_per_step.sum(dim=1) / denom).mean()

    def _superpose_scalar_position_weights(
        self,
        t_len: int,
        token_counts: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        if t_len <= 0:
            return torch.zeros((*token_counts.shape, 0), device=device, dtype=dtype)
        mode = self._resolve_scalar_positional_mode(mode)
        pos_idx = torch.arange(t_len, device=device, dtype=dtype).view(1, 1, t_len)

        if mode == "fb":
            raise ValueError("fb is handled in the caller.")
        if mode == "geometric":
            decay = max(float(getattr(self.config, "ccot_superpose_kl_scalar_positional_decay", 0.9)), 1e-8)
            base = torch.full((1, 1, 1), decay, device=device, dtype=dtype)
            return torch.pow(base, pos_idx)
        if mode == "sinusoidal":
            amp = float(getattr(self.config, "ccot_superpose_kl_scalar_positional_sin_amp", 1.0))
            denom_pos = (token_counts.to(dtype) - 1.0).clamp(min=1.0).unsqueeze(-1)
            pos_norm = pos_idx / denom_pos
            return torch.exp(amp * torch.sin(0.5 * math.pi * pos_norm))
        if mode in {"transformer_sinusoidal", "rope"}:
            amp = float(getattr(self.config, "ccot_superpose_kl_scalar_positional_sin_amp", 1.0))
            num_pairs = max(1, int(getattr(self.config, "ccot_superpose_kl_scalar_positional_fourier_pairs", 16)))
            base = max(float(getattr(self.config, "ccot_superpose_kl_scalar_positional_base", 10000.0)), 1.0001)
            freq_sampling = str(
                getattr(self.config, "ccot_superpose_kl_scalar_positional_freq_sampling", "spread")
            ).strip().lower()
            if freq_sampling not in {"prefix", "spread"}:
                freq_sampling = "spread"
            d_model_ref = int(getattr(self.config, "hidden_size", getattr(self.config, "n_embd", 0)))
            if d_model_ref <= 0:
                d_model_ref = 2 * num_pairs
            d_model_ref = max(d_model_ref, 2 * num_pairs)
            full_pairs = max(1, d_model_ref // 2)
            if num_pairs >= full_pairs:
                pair_idx = torch.arange(full_pairs, device=device, dtype=dtype)
            elif freq_sampling == "prefix":
                pair_idx = torch.arange(num_pairs, device=device, dtype=dtype)
            else:
                pair_idx = torch.linspace(0, full_pairs - 1, steps=num_pairs, device=device, dtype=dtype)
                pair_idx = torch.round(pair_idx)
            inv_freq = torch.exp((-math.log(base) * (2.0 * pair_idx)) / float(d_model_ref))
            angles = pos_idx.unsqueeze(-1) * inv_freq.view(1, 1, 1, num_pairs)
            sin_part = torch.sin(angles)
            cos_part = torch.cos(angles)
            if mode == "transformer_sinusoidal":
                score = (sin_part + cos_part).mean(dim=-1) / math.sqrt(2.0)
            else:
                score = cos_part.mean(dim=-1)
            return torch.exp(amp * score)
        if mode == "learned":
            if self.superpose_kl_scalar_positional_logits is None:
                raise RuntimeError(
                    "Learned scalar positional KL mode selected but position parameter table is not initialized."
                )
            table = self.superpose_kl_scalar_positional_logits
            max_pos = int(table.numel())
            idx = torch.arange(t_len, device=device, dtype=torch.long).clamp(max=max_pos - 1)
            weights_1d = F.softplus(table.to(device=device)[idx]).to(dtype) + torch.tensor(
                1e-6, device=device, dtype=dtype
            )
            return weights_1d.view(1, 1, t_len)
        raise ValueError(f"Unsupported scalar positional KL mode={mode!r}.")

    def _kl_from_superposed_token_targets_scalar_positional(
        self,
        student_logits: torch.Tensor,
        target_token_ids: torch.LongTensor,
        target_token_mask: torch.Tensor,
        step_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        resolved_mode = self._resolve_scalar_positional_mode(mode)
        if resolved_mode == "fb":
            kl_forward = self._kl_from_superposed_token_targets_scalar_positional(
                student_logits=student_logits,
                target_token_ids=target_token_ids,
                target_token_mask=target_token_mask,
                step_mask=step_mask,
                temperature=temperature,
                mode="geometric",
            )
            kl_backward = self._kl_from_superposed_token_targets_scalar_positional(
                student_logits=student_logits,
                target_token_ids=target_token_ids,
                target_token_mask=target_token_mask,
                step_mask=step_mask,
                temperature=temperature,
                mode="geometric_reverse",
            )
            return 0.5 * (kl_forward + kl_backward)

        if student_logits is None:
            raise RuntimeError("student_logits is required for scalar positional superposed-token KL.")
        if target_token_ids is None or target_token_mask is None:
            raise RuntimeError("Scalar positional superposed-token KL requires target step token ids and masks.")
        if student_logits.ndim != 3:
            raise ValueError(f"Expected student_logits to be [B,K,V], got shape={tuple(student_logits.shape)}")
        if target_token_ids.ndim != 3 or target_token_mask.ndim != 3:
            raise ValueError(
                "Expected target_token_ids/target_token_mask to be [B,S,T], "
                f"got {tuple(target_token_ids.shape)} and {tuple(target_token_mask.shape)}"
            )

        device = student_logits.device
        _, num_slots, _ = student_logits.shape
        _, num_steps, _ = target_token_ids.shape
        aligned_steps = min(num_slots, num_steps)
        if aligned_steps <= 0:
            return torch.tensor(0.0, device=device, dtype=student_logits.dtype)

        logits = student_logits[:, :aligned_steps, :]
        if temperature != 1.0:
            logits = logits / float(temperature)
        log_probs = F.log_softmax(logits, dim=-1)

        ids = target_token_ids[:, :aligned_steps, :].to(device)
        token_mask = target_token_mask[:, :aligned_steps, :].to(device=device, dtype=log_probs.dtype)
        token_counts = token_mask.sum(dim=-1)

        if step_mask is not None:
            valid_step_mask = step_mask[:, :aligned_steps].to(device=device, dtype=log_probs.dtype)
            valid_step_mask = valid_step_mask * (token_counts > 0).to(log_probs.dtype)
        else:
            valid_step_mask = (token_counts > 0).to(log_probs.dtype)
        if float(valid_step_mask.sum().detach()) <= 0.0:
            return torch.tensor(0.0, device=device, dtype=student_logits.dtype)

        safe_ids = torch.where(token_mask > 0, ids, torch.zeros_like(ids))
        gathered_log_probs = log_probs.gather(dim=-1, index=safe_ids)

        if resolved_mode == "geometric_reverse":
            decay = max(float(getattr(self.config, "ccot_superpose_kl_scalar_positional_decay", 0.9)), 1e-8)
            base = torch.full((1, 1, 1), decay, device=device, dtype=log_probs.dtype)
            pos_idx = torch.arange(ids.size(-1), device=device, dtype=log_probs.dtype).view(1, 1, -1)
            rev_exp = ((token_counts.to(log_probs.dtype) - 1.0).clamp(min=0.0).unsqueeze(-1) - pos_idx).clamp(min=0.0)
            raw_alpha = torch.pow(base, rev_exp) * token_mask
        else:
            raw_alpha = self._superpose_scalar_position_weights(
                t_len=ids.size(-1),
                token_counts=token_counts,
                dtype=log_probs.dtype,
                device=device,
                mode=resolved_mode,
            ) * token_mask

        alpha_sums = raw_alpha.sum(dim=-1)
        valid_step_mask = valid_step_mask * (alpha_sums > 0).to(log_probs.dtype)
        if float(valid_step_mask.sum().detach()) <= 0.0:
            return torch.tensor(0.0, device=device, dtype=student_logits.dtype)
        alpha = raw_alpha / alpha_sums.clamp(min=1e-12).unsqueeze(-1)

        ce_per_step = -(alpha * gathered_log_probs).sum(dim=-1)
        with torch.no_grad():
            entropy_per_step = self._entropy_from_token_weights(
                token_ids=ids,
                token_weights=alpha,
            )

        kl_per_step = (ce_per_step - entropy_per_step) * valid_step_mask
        denom = valid_step_mask.sum(dim=1).clamp(min=1.0)
        return (kl_per_step.sum(dim=1) / denom).mean()

    def _compute_superpose_kl_loss(
        self,
        student_slot_hidden_states: torch.Tensor,
        target_token_ids: torch.LongTensor,
        target_token_mask: torch.Tensor,
        step_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if student_slot_hidden_states is None:
            raise RuntimeError("student_slot_hidden_states is required for superpose KL.")
        if target_token_ids is None or target_token_mask is None:
            raise RuntimeError("Superpose KL requires target step token ids and masks.")
        if student_slot_hidden_states.ndim != 3:
            raise ValueError(
                "Expected student_slot_hidden_states to be [B,K,D], "
                f"got {tuple(student_slot_hidden_states.shape)}"
            )

        cfg = self.config
        num_slots = int(student_slot_hidden_states.size(1))
        use_uniform_chunk = bool(getattr(cfg, "ccot_superpose_uniform_chunk_to_latents", False))
        use_random_chunk = bool(getattr(cfg, "ccot_superpose_random_chunk_to_latents", False))
        use_random_displace = bool(getattr(cfg, "ccot_superpose_random_displace_steps_to_latents", False))
        use_uniform_token_chunk = bool(getattr(cfg, "ccot_superpose_uniform_token_chunk_to_latents", False))
        if use_uniform_token_chunk:
            target_token_ids, target_token_mask, step_mask = self._chunk_superpose_tokens_to_latents(
                target_token_ids=target_token_ids,
                target_token_mask=target_token_mask,
                step_mask=step_mask,
                num_latent=num_slots,
            )
        elif use_uniform_chunk or use_random_chunk:
            target_token_ids, target_token_mask, step_mask = self._chunk_superpose_steps_to_latents(
                target_token_ids=target_token_ids,
                target_token_mask=target_token_mask,
                step_mask=step_mask,
                num_latent=num_slots,
                randomize=use_random_chunk,
            )
        if use_random_displace and self.training:
            target_token_ids, target_token_mask, step_mask = self._randomly_displace_superpose_steps_to_latents(
                target_token_ids=target_token_ids,
                target_token_mask=target_token_mask,
                step_mask=step_mask,
                num_latent=num_slots,
            )

        student_logits = self.lm_head(student_slot_hidden_states).float()
        temperature = float(getattr(cfg, "ccot_kl_temperature", 1.0))
        use_bag = bool(getattr(cfg, "ccot_superpose_kl_use_bag", True))
        use_scalar = bool(getattr(cfg, "ccot_superpose_kl_scalar_positional", False))
        if not use_bag and not use_scalar:
            raise RuntimeError(
                "ccot_superpose_kl=True but all superpose KL components are disabled "
                "(ccot_superpose_kl_use_bag=False and ccot_superpose_kl_scalar_positional=False)."
            )

        bag_raw = None
        scalar_raw = None
        total_raw = torch.tensor(0.0, device=student_logits.device, dtype=student_logits.dtype)

        if use_bag:
            bag_raw = self._kl_from_superposed_token_targets(
                student_logits=student_logits,
                target_token_ids=target_token_ids,
                target_token_mask=target_token_mask,
                step_mask=step_mask,
                temperature=temperature,
            )
            total_raw = total_raw + bag_raw

        scalar_weight = float(getattr(cfg, "ccot_superpose_kl_scalar_positional_weight", 1.0))
        if use_scalar:
            scalar_raw = self._kl_from_superposed_token_targets_scalar_positional(
                student_logits=student_logits,
                target_token_ids=target_token_ids,
                target_token_mask=target_token_mask,
                step_mask=step_mask,
                temperature=temperature,
                mode=getattr(cfg, "ccot_superpose_kl_scalar_positional_mode", "geometric"),
            )
            total_raw = total_raw + scalar_weight * scalar_raw

        beta = float(getattr(cfg, "ccot_superpose_beta", 1.0))
        loss = beta * total_raw

        logs: Dict[str, float] = {
            "kl_loss": float(loss.detach()),
            "superpose_kl_raw": float(total_raw.detach()),
            "superpose_kl_weighted_raw": float(total_raw.detach()),
            "superpose_kl_weight": 1.0,
            "superpose_beta": beta,
        }
        if bag_raw is not None:
            logs["superpose_kl_bag_raw"] = float(bag_raw.detach())
        if scalar_raw is not None:
            scalar_raw_log = float(scalar_raw.detach())
            logs["superpose_kl_scalar_positional_raw"] = scalar_raw_log
            logs["superpose_kl_scalar_positional_weight"] = scalar_weight
            logs["superpose_kl_scalar_positional_loss"] = beta * scalar_weight * scalar_raw_log
        return loss, logs
