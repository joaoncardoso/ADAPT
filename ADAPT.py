"""
ADAPT: feature visualization for SAE latents.

Usage:
    optimizer = ADAPT(model, sae)                     # sae_lens HookedSAETransformer + SAE
    result = optimizer.optimize(feat_id, OptimizationConfig())
    optimizer.state.prompts, optimizer.state.activations   # final population

The defaults of OptimizationConfig are the settings used in the paper.
"""
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import numpy as np
import sae_lens
import torch
import torch.nn.functional as F

from helpers import SAEFeatureRunner, compute_batch_size, beam_search, group_bests, global_top_k, is_oom

# ==================== #
#        CONFIG        #
# ==================== #

@dataclass
class CullerConfig:
    group_keep: int = 2  # guaranteed slots per group
    merge_point: Optional[int] = None  # after this iteration, groups are ignored and only the best are kept

@dataclass
class EstimatorConfig:
    dist_reg_weight: float = 0.0

@dataclass
class CandidateConfig:
    greedy: Optional[bool] = False
    top_k: int = 512

@dataclass
class DistributionConfig:
    init_method: Literal["uniform", "tail_heavy"] = "uniform"
    update_method: Literal["static", "gradient"] = "static"
    keep_first: bool = True

@dataclass
class InitConfig:
    k1: int = 30  # beam size
    k2: int = 32  # expansion factor
    init_sequences: int = 5000
    group_objectives: Optional[list] = None  # e.g. ["logit", "activation", "activation"]; defaults to all "activation"
    logit_top_k: int = 10

@dataclass
class OptimizationConfig:
    n_iterations: int = 25
    k1: int = 10  # number of parallel prompts
    k2: int = 32  # candidates per prompt
    target_length: int = 10
    n_groups: int = 3
    gcg_prob: float = 0.5
    excess_slots: int = 10  # how many extra free slots to maintain when next operation is logit swap
    culler: CullerConfig = field(default_factory=CullerConfig)
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    candidate: CandidateConfig = field(default_factory=CandidateConfig)
    distribution: DistributionConfig = field(default_factory=DistributionConfig)
    init: InitConfig = field(default_factory=InitConfig)
    apply_ce: bool = True
    ce_penalty_max: float = 3.0
    ce_penalty_threshold: float = 0.0
    ce_penalty_midpoint: int = None  # iteration at steepest increase
    ce_penalty_steepness: float = 0.1  # higher = sharper transition

    sample_k: int = 32

    def __post_init__(self):
        if self.k1 < self.n_groups * self.culler.group_keep:
            raise ValueError("k1 must be at least n_groups * culler.group_keep")
        if not 0 <= self.gcg_prob <= 1:
            raise ValueError("GCG swap probability must be between 0 and 1")
        if self.ce_penalty_midpoint is None:
            self.ce_penalty_midpoint = self.n_iterations // 2
        if self.excess_slots < 0 or self.k1 * self.k2 < self.k1 + self.excess_slots: # prevent having more slots than generated candidates
            raise ValueError("Invalid excess slots size")

@dataclass
class OptimizationResult:
    best_prompt: torch.Tensor
    loss_history: list # best loss history
    history: list # list of optimization states
    config: OptimizationConfig
    runtime: Optional[float]

# ==================== #
#        STATE         #
# ==================== #

@dataclass
class CompactLogits:
    """Only stores top probabilities in order to save memory
    Because of this, ADAPT only works with top-k sampling (rather than top-p or whole vocab sampling)
    """
    top_indices: torch.Tensor  # [batch, seq, k]
    top_probs: torch.Tensor    # [batch, seq, k]
    k: int = 128

    @classmethod
    def from_logits(cls, logits: torch.Tensor, k: int = 128, blacklist: Optional[list] = None):
        # chunked over the batch to bound peak memory (large vocabularies otherwise OOM on the softmax copy)
        tp, ti = [], []
        for c in range(0, logits.shape[0], 64):
            probs = F.softmax(logits[c:c + 64], dim=-1)
            if blacklist is not None:
                probs[:, :, blacklist] = 0
            p_, i_ = torch.topk(probs, k=k, dim=-1)
            tp.append(p_); ti.append(i_)
            del probs
        return cls(torch.cat(ti), torch.cat(tp), k)

    def sample(self, batch_idx: int, pos: int, num_samples: int = 1):
        return torch.multinomial(
            self.top_probs[batch_idx, pos],
            num_samples=num_samples
        )

    def __getitem__(self, indices):
        # in order to make logits subscriptable
        return CompactLogits(
            top_indices=self.top_indices[indices],
            top_probs=self.top_probs[indices],
            k=self.k
        )

@dataclass
class State:
    prompts: torch.Tensor
    groups: torch.Tensor
    masks: torch.Tensor
    logits: CompactLogits
    activations: torch.Tensor
    xentropy: torch.Tensor
    xentropy_weight: float
    target: torch.Tensor
    improvements: Optional[torch.Tensor] = None

class ADAPT:
    def __init__(self, model, sae):
        self.model = model
        self.sae = sae
        self.device = model.cfg.device
        self.history = []
        self.best_loss_history = []

    def optimize(self, feat_id, config: Optional[OptimizationConfig] = None, init_prompts: Optional[torch.Tensor] = None, init_masks: Optional[torch.Tensor] = None, verbose: bool = True) -> OptimizationResult:

        start_time = time.time()
        config = config or OptimizationConfig()
        self.history = []
        self.best_loss_history = []

        self.runner = SAEFeatureRunner(self.model, self.sae, feat_id)

        if init_prompts is not None:
            n_prompts = init_prompts.shape[0]
            if n_prompts != config.k1:
                raise ValueError(f"Expected {config.k1} prompts, got {n_prompts}")
            # Round-robin group assignment
            init_groups = torch.arange(n_prompts, device=init_prompts.device) % config.n_groups

            if init_masks is None:
                pad_token_id = self.model.tokenizer.pad_token_id or self.model.tokenizer.eos_token_id
                init_masks = (init_prompts != pad_token_id).long()
        else:
            init_prompts, init_masks, init_groups = self.dbs_init(config, self.model, self.sae, feat_id, verbose)

        # special ids outside the embedding matrix (e.g. Gemma 3's image token) can't be produced anyway
        blacklist = [i for i in self.model.tokenizer.all_special_ids if i < self.model.cfg.d_vocab]

        # initialize estimator, evaluator, culler, distribution_handler, logit_generator, gcg_generator, etc.
        estimator = self._create_estimator(config, self.runner)
        gcg_candidate_generator, logit_candidate_generator = self._create_candidate_generators(config, self.device)
        evaluator = self._create_evaluator(config, self.runner, blacklist)
        culler = self._create_culler(config)
        distribution_handler = self._create_distribution_handler(config)

        self.state = evaluator.evaluate(init_prompts, init_masks, init_groups, current_iteration=0)

        op_probs = torch.tensor([config.gcg_prob, 1 - config.gcg_prob], device=self.device)  # 0 is gcg swap, 1 is logit swap

        n_iterations = config.n_iterations

        def sample_op():
            operation_idx = torch.multinomial(op_probs, 1).item()
            operation = ["gcg_swap", "logit_swap"][operation_idx]
            return operation

        current_op = sample_op()
        next_op = sample_op()

        # Main loop
        for i in range(n_iterations):

            if verbose:
                print(f"Iteration {i+1}/{n_iterations}")
                if config.apply_ce:
                    print(f"Current ce penalty weight: {self.state.xentropy_weight}")
                # print current state
                for p in range(self.state.prompts.shape[0]):
                    decoded = self.model.tokenizer.decode(self.state.prompts[p])
                    decoded = decoded.replace("\n", "＼n")  # fullwidth reverse solidus
                    decoded = repr(decoded)[1:-1]
                    # Get next most likely token
                    next_token_idx = self.state.logits.top_indices[p, -1, 0]  # [batch, seq, k] -> first of top-k
                    next_token_str = self.model.tokenizer.decode(next_token_idx)
                    print(f"  [{p}] Group {self.state.groups[p].item()} | Score: {self.state.target[p].item():.2f} | Len: {self.state.masks[p].sum().item()} | ce penalty: {self.state.xentropy[p].item():.4f} | Act: {self.state.activations[p].item():.2f} | {decoded}[{next_token_str}]")

            # if GCG, compute estimate
            if current_op == "gcg_swap":
                estimate = estimator.estimate(self.state, i)
                estimate_bounds = {}
                for p in range(estimate.shape[0]):  # num_prompts
                    for pos in range(estimate.shape[1]):  # seq_len
                        estimate_bounds[(p, pos)] = {
                            "min": estimate[p, pos].min().item(),
                        }
                if config.distribution.update_method == "gradient":
                    distribution_handler.update(estimate_bounds, self.state.masks)
                elif config.distribution.update_method == "static":
                    distribution_handler.update(self.state.masks)

                candidate_ids, candidate_masks, candidate_groups, mutation_positions, candidate_estimates = gcg_candidate_generator.generate(
                    self.state, estimate, distribution_handler.dist, blacklist
                )
                del estimate

            elif current_op == "logit_swap":
                # logit swap should become logit insert for underlength sequences
                candidate_ids, candidate_masks, candidate_groups, mutation_positions = logit_candidate_generator.generate(
                    self.state
                )  # don't pass distribution

            # evaluator
            candidate_state = evaluator.evaluate(
                candidate_ids, candidate_masks, candidate_groups, i, self.state, mutation_positions
            )

            if current_op == "gcg_swap":
                new_state, improvement_gradients = culler.cull(
                    candidate_state, self.state, next_op, candidate_estimates,
                )
                targets = new_state.target
                self.history.append(self.HistoryRecord("gcg_swap", targets, self.state, new_state.prompts, improvement_gradients, estimate_bounds, distribution_handler.dist, self.model.to_string(new_state.prompts)))

            elif current_op == "logit_swap":
                new_state, _ = culler.cull(candidate_state, self.state, next_op)
                targets = new_state.target
                self.history.append(self.HistoryRecord("logit_swap", targets, self.state, new_state.prompts, None, None, None, self.model.to_string(new_state.prompts)))
            if verbose:
                if config.apply_ce:
                    print(f"  Best score: {targets.max().item():.4f} | {new_state.activations.max().item()}, {new_state.xentropy[new_state.target.argmax().item()]} | Best prompt: {self.model.to_string(new_state.prompts[new_state.target.argmax()])}")
                else:
                    print(f"  Best score: {targets.max().item():.4f} | Best prompt: {self.model.to_string(new_state.prompts[new_state.target.argmax()])}")
            self.best_loss_history.append(targets.max().item())
            self.state = new_state

            current_op = next_op
            next_op = sample_op()

            del candidate_ids
            torch.cuda.empty_cache()
        return OptimizationResult(
            best_prompt=self.state.prompts[self.state.target.argmax()],
            loss_history=self.best_loss_history,
            history=self.history,
            config=config,
            runtime=time.time() - start_time,
        )

# ==================== #
#       HELPERS        #
# ==================== #

    @dataclass
    class HistoryRecord:
        """History record with numpy arrays for serialization"""
        op: Literal["gcg_swap", "logit_swap"]
        losses: np.ndarray
        state: State  # Keep State as-is for runtime access
        prompts: np.ndarray
        improvement_gradients: list
        estimate_bounds: dict
        dist: np.ndarray
        decoded_prompts: list[str]

        def __init__(self, op, losses, state, prompts, improvement_gradients,
                     estimate_bounds, dist, decoded_prompts):
            self.op = op
            # Convert tensors to numpy
            self.losses = losses.cpu().numpy() if torch.is_tensor(losses) else losses
            self.state = state  # Keep as-is for runtime access
            self.prompts = prompts.cpu().numpy() if torch.is_tensor(prompts) else prompts
            self.improvement_gradients = improvement_gradients
            self.estimate_bounds = estimate_bounds
            self.dist = dist.cpu().numpy() if torch.is_tensor(dist) else (dist if dist is not None else None)
            self.decoded_prompts = decoded_prompts

    def dbs_init(self, config: OptimizationConfig, model, sae, feat_id, verbose=True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Initialize ADAPT using diverse beam search.

            Guaranteed slots: Best sequence per group (using each group's own objective).
            Free slots: Top sequences by activation score across entire history.
            """
            init_config = config.init
            n_groups = config.n_groups

            assert init_config.k1 % n_groups == 0, f"init config k1 must be a multiple of the number of groups"

            # Build group objectives list from config (default to activation if not specified)
            group_objectives = init_config.group_objectives
            if group_objectives is None:
                group_objectives = ["activation"] * n_groups

            if verbose:
                print(f"Initializing ADAPT with beam search:")
                print(f"  - Beam size: {init_config.k1} x {init_config.k2}")
                print(f"  - Groups: {n_groups}")
                print(f"  - Group objectives: {group_objectives}")
                print(f"  - Target length: {config.target_length}")

            history = beam_search(
                model, sae, feat_id,
                beam_size=init_config.k1,
                expansion_factor=init_config.k2,
                max_length=config.target_length,
                n_groups=n_groups,
                group_objectives=group_objectives,
                logit_top_k=init_config.logit_top_k,
                init_seq_size=init_config.init_sequences,
                verbose=verbose,
            )

            # === Extract best sequences from history ===

            # Get per-group bests (each using its own objective)
            bests = group_bests(history, group_objectives)

            # Get global top-k by activation score (for free slots)
            # Request more than we need to have enough after deduplication with group bests
            top_k = global_top_k(history, k=config.k1 * 2)

            # === Build final sequence list ===

            guaranteed_slots = n_groups * config.culler.group_keep
            remaining_slots = config.k1 - guaranteed_slots

            final_sequences = []
            final_masks = []
            final_groups = []
            used_sequences = set()  # Track sequences we've already added (as tuples for hashing)

            pad_token_id = model.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = model.tokenizer.eos_token_id

            def pad_sequence(seq, target_length):
                """Left-pad sequence to target length, return (padded_seq, attention_mask)"""
                current_length = seq.shape[0]

                if current_length < target_length:
                    pad_length = target_length - current_length
                    padded_seq = torch.cat([
                        torch.full((pad_length,), pad_token_id, dtype=seq.dtype, device=seq.device),
                        seq
                    ])
                    attention_mask = torch.cat([
                        torch.zeros(pad_length, dtype=torch.long, device=seq.device),
                        torch.ones(current_length, dtype=torch.long, device=seq.device)
                    ])
                else:
                    padded_seq = seq[:target_length]
                    attention_mask = torch.ones(target_length, dtype=torch.long, device=seq.device)

                return padded_seq, attention_mask

            # Phase 1: Add group_keep sequences from each group's best (using their own objective)
            if verbose:
                print("\n  Guaranteed slots (per-group bests):")
            for group_id in range(n_groups):
                if group_id not in bests:
                    raise ValueError(f"No best sequence found for group {group_id}")

                best_seq, best_score, objective = bests[group_id]
                seq_tuple = tuple(best_seq.tolist())

                for copy_idx in range(config.culler.group_keep):
                    padded_seq, attention_mask = pad_sequence(best_seq, config.target_length)
                    final_sequences.append(padded_seq)
                    final_masks.append(attention_mask)
                    final_groups.append(group_id)

                used_sequences.add(seq_tuple)

                if verbose:
                    decoded = model.tokenizer.decode(best_seq)
                    print(f"    Group {group_id} [{objective}]: {best_score:.3f} x{config.culler.group_keep} | {repr(decoded[:60])}")

            # Phase 2: Fill remaining slots with global top-k by activation (skipping already-used sequences)
            added_free = 0

            for seq, score, gid in top_k:
                if added_free >= remaining_slots:
                    break

                seq_tuple = tuple(seq.tolist())
                if seq_tuple in used_sequences:
                    continue  # Skip duplicates

                used_sequences.add(seq_tuple)

                padded_seq, attention_mask = pad_sequence(seq, config.target_length)
                final_sequences.append(padded_seq)
                final_masks.append(attention_mask)
                final_groups.append(gid)  # Use original group from history
                added_free += 1

            if verbose:
                print(f"\n  Free slots: added {added_free}/{remaining_slots} sequences from global top-k (activation)")

            if added_free < remaining_slots:
                print(f"    WARNING: Only found {added_free}/{remaining_slots} unique sequences for free slots")

            # Convert to tensors
            device = model.cfg.device
            init_prompts = torch.stack(final_sequences).to(device)
            init_masks = torch.stack(final_masks).to(device)
            init_groups = torch.tensor(final_groups, dtype=torch.long, device=device)

            # Validate
            assert init_prompts.shape[0] == config.k1, f"Expected {config.k1} sequences, got {init_prompts.shape[0]}"
            unique_groups = torch.unique(init_groups)
            assert len(unique_groups) == n_groups, f"Expected {n_groups} groups, got {len(unique_groups)}"

            return init_prompts, init_masks, init_groups

# ==================== #
#        CULLER        #
# ==================== #

    class Culler(ABC):
        @abstractmethod
        def cull():
            pass

    class GroupCuller(Culler):
        def __init__(self, config):
            self.group_keep = config.culler.group_keep
            # after this iteration, we only maintain the best
            self.merge_point = config.culler.merge_point
            self.k1 = config.k1
            self.excess_slots = config.excess_slots
            self.current_iteration = 0

        def cull(self, candidate_state, prev_state, next_operation, candidate_estimates=None, iteration=None):
            if iteration is not None:
                self.current_iteration = iteration

            # Check if we should merge (switch to pure global selection)
            if self.merge_point is not None and self.current_iteration >= self.merge_point:
                return self._global_cull(candidate_state, prev_state, next_operation)
            else:
                return self._group_balanced_cull(candidate_state, prev_state, candidate_estimates, next_operation)

        def _group_balanced_cull(self, candidate_state, prev_candidate_state, candidate_estimates, next_operation):
            device = candidate_state.prompts.device

            # Combine current and previous states for group-wise selection
            combined_prompts = torch.cat([prev_candidate_state.prompts, candidate_state.prompts], dim=0)
            combined_groups = torch.cat([prev_candidate_state.groups, candidate_state.groups], dim=0)
            combined_masks = torch.cat([prev_candidate_state.masks, candidate_state.masks], dim=0)
            combined_logits = CompactLogits(
                top_indices=torch.cat([prev_candidate_state.logits.top_indices, candidate_state.logits.top_indices], dim=0),
                top_probs=torch.cat([prev_candidate_state.logits.top_probs, candidate_state.logits.top_probs], dim=0),
                k=candidate_state.logits.k
            )
            combined_activations = torch.cat([prev_candidate_state.activations, candidate_state.activations], dim=0)
            combined_xentropy = torch.cat([prev_candidate_state.xentropy, candidate_state.xentropy], dim=0)
            combined_target = torch.cat([prev_candidate_state.target, candidate_state.target], dim=0)

            unique_groups = torch.unique(combined_groups)
            n_groups = len(unique_groups)

            # Calculate remaining slots after group guarantees
            guaranteed_slots = n_groups * self.group_keep
            if next_operation == "gcg_swap":
                remaining_slots = self.k1 - guaranteed_slots
            elif next_operation == "logit_swap":
                remaining_slots = self.k1 - guaranteed_slots + self.excess_slots

            if remaining_slots < 0:
                raise ValueError(f"k1 ({self.k1}) < n_groups * group_keep ({guaranteed_slots})")

            new_candidate_start_idx = prev_candidate_state.prompts.shape[0]
            selected_indices = []

            # Phase 1: Select best group_keep from each group based on ACTIVATIONS (not target)
            # This ensures selection is based on actual activation improvement
            for group_id in unique_groups:
                group_mask = combined_groups == group_id
                group_indices = torch.where(group_mask)[0]

                if len(group_indices) == 0:
                    continue

                # Use activations for ranking (not target which includes CE penalty)
                group_activations = combined_activations[group_indices]
                _, sorted_within_group = torch.topk(group_activations, len(group_indices), largest=True)
                sorted_group_indices = group_indices[sorted_within_group]

                group_selected = []
                used_prev_state = False

                # Go through candidates in order of activation, filling group slots
                for idx in sorted_group_indices:
                    if len(group_selected) >= self.group_keep:
                        break

                    idx_item = idx.item()
                    is_from_prev = idx_item < new_candidate_start_idx

                    # If this is from prev_state and we already used one from prev_state, skip
                    if is_from_prev and used_prev_state:
                        continue

                    # Add this candidate
                    group_selected.append(idx_item)

                    # Mark that we've used prev_state if this was from prev_state
                    if is_from_prev:
                        used_prev_state = True

                selected_indices.extend(group_selected)

            # Phase 2: Global selection from remaining NEW candidates based on activation improvements
            # with per-group cap of 2x expected allotment
            if remaining_slots > 0:
                selected_set = set(selected_indices)

                # Calculate per-group cap for free slots
                expected_allotment_per_group = remaining_slots // n_groups
                max_free_slots_per_group = expected_allotment_per_group * 2

                # Track how many free slots each group has used
                group_free_slot_counts = {g.item(): 0 for g in unique_groups}

                # Only consider NEW candidates (from candidate_state) for remaining slots
                remaining_new_candidates = [
                    i for i in range(new_candidate_start_idx, combined_prompts.shape[0])
                    if i not in selected_set
                ]

                if len(remaining_new_candidates) > 0:
                    # Build list of (combined_idx, target_score, group_id)
                    # Use target (includes CE penalty) for free slot selection
                    candidate_data = []
                    for combined_idx in remaining_new_candidates:
                        target_score = combined_target[combined_idx].item()
                        group_id = combined_groups[combined_idx].item()
                        candidate_data.append((combined_idx, target_score, group_id))

                    # Sort by target score (descending) - CE penalty influences ranking here
                    candidate_data.sort(key=lambda x: x[1], reverse=True)

                    # Greedily select candidates respecting per-group cap
                    additional_selected = []
                    for combined_idx, improvement, group_id in candidate_data:
                        if len(additional_selected) >= remaining_slots:
                            break

                        # Check if this group has room
                        if group_free_slot_counts[group_id] < max_free_slots_per_group:
                            additional_selected.append(combined_idx)
                            group_free_slot_counts[group_id] += 1

                    selected_indices.extend(additional_selected)

            # Create new state with selected candidates
            selected_indices = torch.tensor(selected_indices, device=device)

            return State(
                prompts=combined_prompts[selected_indices],
                groups=combined_groups[selected_indices],
                masks=combined_masks[selected_indices],
                logits=CompactLogits(
                    top_indices=combined_logits.top_indices[selected_indices],
                    top_probs=combined_logits.top_probs[selected_indices],
                    k=combined_logits.k
                ),
                activations=combined_activations[selected_indices],
                xentropy=combined_xentropy[selected_indices],
                xentropy_weight=candidate_state.xentropy_weight,
                target=combined_target[selected_indices]
            ), None  # No improvement_gradients for now

        def _global_cull(self, candidate_state, prev_candidate_state, next_operation):
            """Pure global selection after merge point - based on activations"""
            # Combine both states for global selection
            combined_prompts = torch.cat([prev_candidate_state.prompts, candidate_state.prompts], dim=0)
            combined_groups = torch.cat([prev_candidate_state.groups, candidate_state.groups], dim=0)
            combined_masks = torch.cat([prev_candidate_state.masks, candidate_state.masks], dim=0)
            combined_logits = CompactLogits(
                top_indices=torch.cat([prev_candidate_state.logits.top_indices, candidate_state.logits.top_indices], dim=0),
                top_probs=torch.cat([prev_candidate_state.logits.top_probs, candidate_state.logits.top_probs], dim=0),
                k=candidate_state.logits.k
            )
            combined_activations = torch.cat([prev_candidate_state.activations, candidate_state.activations], dim=0)
            combined_xentropy = torch.cat([prev_candidate_state.xentropy, candidate_state.xentropy], dim=0)
            combined_target = torch.cat([prev_candidate_state.target, candidate_state.target], dim=0)

            # Select based on activations (not target)
            if next_operation == "gcg_swap":
                _, top_indices = torch.topk(combined_activations, self.k1, largest=True)
            elif next_operation == "logit_swap":
                _, top_indices = torch.topk(combined_activations, self.k1 + self.excess_slots, largest = True)
            else:
                raise ValueError("Must pass next_iteration to culler")

            return State(
                prompts=combined_prompts[top_indices],
                groups=combined_groups[top_indices],  # Keep group tracking but don't use it
                masks=combined_masks[top_indices],
                logits=CompactLogits(
                    top_indices=combined_logits.top_indices[top_indices],
                    top_probs=combined_logits.top_probs[top_indices],
                    k=combined_logits.k
                ),
                activations=combined_activations[top_indices],
                xentropy=combined_xentropy[top_indices],
                xentropy_weight=candidate_state.xentropy_weight,
                target=combined_target[top_indices]
            ), None

    def _create_culler(self, config):
        """Factory method for creating culler"""
        return self.GroupCuller(config)

# ==================== #
#      CANDIDATES      #
# ==================== #

    class CandidateGenerator(ABC):
        def __init__(self, config: OptimizationConfig, device):
            self.k2 = config.k2
            self.device = device

        @abstractmethod
        def generate(self):
            pass

    class LogitSwapGenerator(CandidateGenerator):
        """Simple logit-based token swapping and insertion"""

        def __init__(self, config: OptimizationConfig, device):
            self.target_length = config.target_length
            self.sample_k = config.sample_k
            self.keep_first = config.distribution.keep_first
            super().__init__(config, device)

        def generate(self, state: State) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Generate candidates using logit swap/insert operations

            Returns:
                Tuple of (candidate_prompts, candidate_masks, candidate_groups, mutation_positions)
            """

            device = self.device
            k1 = state.prompts.shape[0]
            k2 = self.k2
            total_candidates = k1 * k2
            seq_len = state.prompts.shape[1]

            # Pre-allocate outputs
            candidate_prompts = torch.zeros((total_candidates, seq_len), dtype=state.prompts.dtype, device=device)
            candidate_masks = torch.zeros((total_candidates, seq_len), dtype=torch.long, device=device)
            candidate_groups = torch.zeros(total_candidates, dtype=state.groups.dtype, device=device)
            mutation_positions = torch.zeros(total_candidates, dtype=torch.long, device=device)

            cand_idx = 0

            # Loop through each original prompt
            for i in range(k1):
                # Get actual length of this prompt (sum of attention mask)
                actual_length = state.masks[i].sum().item()
                used_mutations = set()  # Track (position, token) pairs to avoid duplicates

                # Loop for k2 candidates from this prompt
                for child_idx in range(k2):
                    # Copy original prompt and group
                    candidate_prompts[cand_idx] = state.prompts[i]
                    candidate_masks[cand_idx] = state.masks[i]
                    candidate_groups[cand_idx] = state.groups[i]

                    # Try to find a unique mutation
                    attempts = 0
                    max_attempts = 50  # Reasonable limit to avoid infinite loops

                    while attempts < max_attempts:
                        if actual_length < self.target_length:
                            # INSERT: prompt is underlength
                            start_pos = (state.masks[i] == 0).sum().item()  # Number of padding tokens
                            insertion_start = start_pos + (1 if self.keep_first else 0)  # Skip BOS if keep_first
                            pos = torch.randint(insertion_start, start_pos + actual_length, (1,), device=device).item()

                            # Sample token from logits at position before insertion
                            logit_pos = max(0, pos - 1)
                            sampled_indices = state.logits.sample(i, logit_pos, num_samples=self.sample_k)
                            # Use different sample for each child to add variety
                            token_choice_idx = (child_idx + attempts) % self.sample_k
                            selected_token = state.logits.top_indices[i, logit_pos, sampled_indices[token_choice_idx]]

                        else:
                            # SWAP: prompt has target length
                            pos = torch.randint(1, actual_length, (1,), device=device).item()

                            # Sample token from logits at PREVIOUS position (predicts current position)
                            sampled_indices = state.logits.sample(i, pos-1, num_samples=self.sample_k)
                            token_choice_idx = (child_idx + attempts) % self.sample_k
                            selected_token = state.logits.top_indices[i, pos, sampled_indices[token_choice_idx]]

                        # Check if this mutation is unique
                        mutation_key = (pos, selected_token.item())
                        if mutation_key not in used_mutations:
                            used_mutations.add(mutation_key)
                            mutation_positions[cand_idx] = pos

                            if actual_length < self.target_length:
                                # INSERT: shift everything left to consume padding
                                candidate_prompts[cand_idx, :pos] = state.prompts[i, 1:pos+1]  # Shift left
                                candidate_prompts[cand_idx, pos] = selected_token              # Insert new token
                                candidate_prompts[cand_idx, pos+1:] = state.prompts[i, pos+1:] # Keep rest the same

                                # Update attention mask: convert leftmost 0 to 1
                                leftmost_zero = 0
                                while leftmost_zero < seq_len and candidate_masks[cand_idx, leftmost_zero] == 1:
                                    leftmost_zero += 1
                                if leftmost_zero < seq_len:
                                    candidate_masks[cand_idx, leftmost_zero] = 1
                            else:
                                # SWAP: replace token at position
                                candidate_prompts[cand_idx, pos] = selected_token

                            break  # Found unique mutation, exit retry loop

                        attempts += 1

                    # If we couldn't find unique mutation after max_attempts, just use the last one
                    # This prevents infinite loops in edge cases
                    if attempts == max_attempts:
                        mutation_positions[cand_idx] = pos
                        if actual_length < self.target_length:
                            candidate_prompts[cand_idx, pos+1:actual_length+1] = state.prompts[i, pos:actual_length]
                            candidate_prompts[cand_idx, pos] = selected_token
                            rightmost_zero = seq_len - 1
                            while rightmost_zero >= 0 and candidate_masks[cand_idx, rightmost_zero] == 1:
                                rightmost_zero -= 1
                            if rightmost_zero >= 0:
                                candidate_masks[cand_idx, rightmost_zero] = 1
                        else:
                            candidate_prompts[cand_idx, pos] = selected_token

                    cand_idx += 1

            return candidate_prompts, candidate_masks, candidate_groups, mutation_positions

    class GCGSwapGenerator(CandidateGenerator):
        def __init__(self, config, device):
            super().__init__(config, device)
            self.greedy = config.candidate.greedy
            self.top_k = config.candidate.top_k

        def generate(self, state: State, estimate: torch.Tensor, dist: torch.Tensor, blacklist: list):
            prompts = state.prompts
            n_prompts, seq_len = prompts.shape

            candidates = prompts.unsqueeze(1).repeat(1, self.k2, 1)
            mutation_positions = torch.zeros((n_prompts, self.k2), dtype=torch.long, device=prompts.device)
            candidate_estimates = torch.zeros((n_prompts, self.k2), dtype=estimate.dtype, device=prompts.device)
            candidate_rankings = torch.zeros((n_prompts, self.k2), dtype=torch.int16, device=prompts.device)
            candidate_masks = state.masks.unsqueeze(1).repeat(1, self.k2, 1)
            candidate_groups = state.groups.unsqueeze(1).repeat(1, self.k2)

            blacklist_tensor = torch.tensor(blacklist, device=estimate.device, dtype=torch.long)
            estimate[:, :, blacklist_tensor] = float('inf')

            all_sampled_positions = torch.zeros((n_prompts, self.k2), dtype=torch.long, device=prompts.device)
            for p in range(n_prompts):
                all_sampled_positions[p] = torch.multinomial(dist[p], self.k2, replacement=True)

            for p in range(n_prompts):
                positions = all_sampled_positions[p]
                used_mutations = set()  # Track (position, token) pairs to avoid exact duplicates

                for m_idx, pos in enumerate(positions):
                    mutation_positions[p, m_idx] = pos
                    token_estimates = estimate[p, pos]
                    _, top_indices = torch.topk(token_estimates, k=self.top_k, largest=False)

                    # Try to find a unique mutation
                    attempts = 0
                    while attempts < self.top_k:
                        if self.greedy:
                            top_estimates = token_estimates[top_indices]
                            weights = torch.abs(top_estimates)
                            weights = weights / weights.sum()
                            selected_idx = torch.multinomial(weights, 1).item()
                        else:
                            selected_idx = torch.randint(0, self.top_k, (1,)).item()

                        new_token = top_indices[selected_idx]
                        mutation_key = (pos.item(), new_token.item())

                        if mutation_key not in used_mutations:
                            used_mutations.add(mutation_key)
                            candidates[p, m_idx, pos] = new_token
                            candidate_estimates[p, m_idx] = token_estimates[new_token]
                            candidate_rankings[p, m_idx] = selected_idx
                            break

                        attempts += 1

                    # If we couldn't find a unique mutation after trying all top_k, just use the last one
                    # (This should be rare with proper top_k size)
                    if attempts == self.top_k:
                        candidates[p, m_idx, pos] = new_token
                        candidate_estimates[p, m_idx] = token_estimates[new_token]
                        candidate_rankings[p, m_idx] = selected_idx

            # Flatten the dimensions from (n_prompts, k2, seq_len) to (n_prompts * k2, seq_len)
            candidates = candidates.view(n_prompts * self.k2, seq_len)
            candidate_masks = candidate_masks.view(n_prompts * self.k2, seq_len)
            candidate_groups = candidate_groups.view(n_prompts * self.k2)
            mutation_positions = mutation_positions.view(n_prompts * self.k2)
            candidate_estimates = candidate_estimates.view(n_prompts * self.k2)

            return candidates, candidate_masks, candidate_groups, mutation_positions, candidate_estimates

    def _create_candidate_generators(self, config, device):
        """Factory method for creating candidate generators"""
        gcg_generator = self.GCGSwapGenerator(config, device)
        logit_generator = self.LogitSwapGenerator(config, device)
        return gcg_generator, logit_generator

    # ==================== #
    #      ESTIMATORS      #
    # ==================== #

    class Estimator(ABC):
        @abstractmethod
        def estimate(self, state):
            pass

    class GCGEstimator(Estimator):
        def __init__(self, config: OptimizationConfig, runner: SAEFeatureRunner):
            self.runner = runner
            self.model = runner.model
            self.dist_reg_weight = config.estimator.dist_reg_weight
            self.apply_ce = config.apply_ce
            self.config = config

        def estimate(self, state, current_iteration):
            prompts = state.prompts
            masks = state.masks
            vocab_size = self.model.W_E.shape[0]
            one_hot = F.one_hot(prompts, num_classes=vocab_size).to(self.model.W_E.dtype)

            one_hot.requires_grad_(True)
            embeddings = torch.matmul(one_hot, self.model.W_E)

            result = self.runner(inputs_embeds=embeddings, attention_mask=masks)
            targets = result["target"]

            loss = -targets.sum()
            # loss doesn't include cross_entropy, only selection does

            loss.backward()

            estimate = one_hot.grad.clone().detach()

            if self.dist_reg_weight > 0:
                with torch.no_grad():
                    batch_size, seq_len, embed_dim = embeddings.shape
                    current_flat = embeddings.view(-1, embed_dim)
                    distances_flat = torch.cdist(current_flat, self.model.W_E)
                    distances = distances_flat.view(batch_size, seq_len, vocab_size)
                    estimate = estimate + self.dist_reg_weight * distances
                    del distances, current_flat, distances_flat

            del loss, targets, result, embeddings, one_hot
            return estimate

    def _create_estimator(self, config, runner):
        """Factory method for creating estimator"""
        return self.GCGEstimator(config, runner)

    # ==================== #
    #       EVALUATOR      #
    # ==================== #

    class Evaluator(ABC):
        @abstractmethod
        def evaluate(self, prompt_ids, prompt_masks, prompt_groups, mutation_positions):
            pass

    class SingleEvaluator(Evaluator):
        def __init__(self, config: OptimizationConfig, runner: SAEFeatureRunner, model: sae_lens.HookedSAETransformer, blacklist: list):
            self.apply_ce = config.apply_ce
            self.config = config
            self.runner = runner
            self.model = model
            self.device = self.model.cfg.device
            self.blacklist = blacklist
            self.batch_cap = None  # lowered after an out-of-memory error

        def _evaluate_batch(self, batch_candidates, batch_masks):
            """Activation, logits and mean self-cross-entropy for one batch"""
            hook_name = f"{self.runner.sae.cfg.metadata.hook_name}.hook_sae_acts_pre"
            batch_logits, cache = self.model.run_with_cache_with_saes(
                batch_candidates,
                attention_mask=batch_masks,
                saes=[self.runner.sae],
                use_error_term=True,
                names_filter=hook_name,
            )
            acts = cache[hook_name][:, -1, self.runner.feature_idxs]

            logits_offset = batch_logits[:, :-1]
            targets = batch_candidates[:, 1:]
            mask = batch_masks[:, 1:]

            token_losses = F.cross_entropy(
                logits_offset.reshape(-1, logits_offset.shape[-1]),
                targets.reshape(-1),
                reduction='none'
            ).reshape(logits_offset.shape[0], -1)

            masked_loss = token_losses * mask
            seq_lengths = mask.sum(dim=1).clamp(min=1)
            batch_xentropy = masked_loss.sum(dim=1) / seq_lengths
            return batch_logits.to(torch.float16), acts.to(torch.float16), batch_xentropy.to(torch.float16)

        @torch.no_grad()
        def evaluate(self, prompt_ids, prompt_masks, prompt_groups, current_iteration, prev_state=None, mutation_positions=None):
            n_prompts, seq_len = prompt_ids.shape
            device = prompt_ids.device

            final_logits = torch.empty((n_prompts, seq_len, self.model.cfg.d_vocab), dtype=torch.float16, device=device)
            final_scores = torch.empty((n_prompts,), dtype=torch.float16, device=device)
            final_xentropy = torch.empty((n_prompts,), dtype=torch.float16, device=device)

            self.model.zero_grad()
            torch.cuda.empty_cache()

            i = 0
            while i < n_prompts:
                # Recompute batch size each iteration for robustness
                batch_size = compute_batch_size(self.model, self.runner.sae, prompt_length=seq_len, device=device)
                if self.batch_cap is not None:
                    batch_size = min(batch_size, self.batch_cap)
                batch_end = min(i + batch_size, n_prompts)

                try:
                    final_logits[i:batch_end], final_scores[i:batch_end], final_xentropy[i:batch_end] = \
                        self._evaluate_batch(prompt_ids[i:batch_end], prompt_masks[i:batch_end])
                except Exception as e:
                    if not is_oom(e) or batch_size == 1:
                        raise
                    # retry this batch with half the size; the failed batch's tensors are freed with its frame
                    self.batch_cap = max(1, batch_size // 2)
                    print(f"OOM at batch {i} (size {batch_size}), retrying with batch size {self.batch_cap}")
                else:
                    i = batch_end
                torch.cuda.empty_cache()

            compact_logits = CompactLogits.from_logits(final_logits, blacklist=self.blacklist)

            # Compute improvements based on ACTIVATIONS (not target)
            # This ensures selection is based on actual activation improvement
            improvements = torch.zeros(n_prompts, dtype=torch.float16, device=device)
            if prev_state is not None:
                # Get best activation per group from previous state
                group_baselines = {}
                for group_id in torch.unique(prev_state.groups):
                    group_mask = prev_state.groups == group_id
                    if group_mask.any():
                        group_baselines[group_id.item()] = prev_state.activations[group_mask].max().item()

                # Calculate improvements based on raw activations
                for group_id in torch.unique(prompt_groups):
                    group_mask = prompt_groups == group_id
                    if group_mask.any() and group_id.item() in group_baselines:
                        baseline = group_baselines[group_id.item()]
                        improvements[group_mask] = final_scores[group_mask] - baseline

            # Apply CE penalty with simple sigmoid schedule (no conditional logic)
            if self.apply_ce:
                # Calculate current penalty using sigmoid schedule
                progress = (current_iteration - self.config.ce_penalty_midpoint) * self.config.ce_penalty_steepness
                current_penalty = self.config.ce_penalty_max / (1 + math.exp(-progress))

                # Apply penalty uniformly to all candidates
                ce_penalty_term = torch.clamp(final_xentropy - self.config.ce_penalty_threshold, min=0)
                target = final_scores - current_penalty * ce_penalty_term
            else:
                # No CE penalty at all
                current_penalty = 0
                target = final_scores

            del final_logits

            return State(
                prompts=prompt_ids,
                groups=prompt_groups,
                masks=prompt_masks,
                logits=compact_logits,
                activations=final_scores,
                target=target,
                xentropy=final_xentropy,
                xentropy_weight=current_penalty,
                improvements=improvements
            )

    def _create_evaluator(self, config, runner, blacklist):
        """Factory method for creating evaluator"""
        return self.SingleEvaluator(config, runner, self.model, blacklist)

    # ==================== #
    #     DISTRIBUTION     #
    # ==================== #

    class DistributionHandler(ABC):
        def __init__(self, config):
            self.init_method = config.distribution.init_method
            self.keep_first = config.distribution.keep_first
            self.dist = None

        @abstractmethod
        def _compute_distribution(self, masks):
            pass

        @abstractmethod
        def update(self, masks):
            pass

    class StaticHandler(DistributionHandler):
        def _compute_distribution(self, masks):
            """Compute distribution based on actual sequence lengths from masks"""
            n_prompts = masks.shape[0]
            max_seq_len = masks.shape[1]

            # Get actual lengths and start positions for each prompt
            seq_lengths = masks.sum(dim=1)  # [n_prompts]
            device = masks.device
            # Create distribution tensor
            dist = torch.zeros((n_prompts, max_seq_len), device=device)

            for i in range(n_prompts):
                actual_len = seq_lengths[i].item()
                if actual_len > 0:
                    # Find where real tokens start (after left padding)
                    start_pos = max_seq_len - actual_len

                    if self.init_method == "uniform":
                        # Uniform distribution over actual tokens
                        weights = torch.ones(actual_len, device=device)
                    elif self.init_method == "tail_heavy":
                        # Flat for first (actual_len - 3), exponential in last 3
                        weights = torch.ones(actual_len, device=device)
                        if actual_len >= 3:
                            ramp_start = actual_len - 3
                            exp_weights = torch.exp(torch.linspace(0.5, 2.0, 3, device=device))
                            weights[ramp_start:] = exp_weights
                    else:
                        raise ValueError(f"Unknown init_method: {self.init_method}")

                    if self.keep_first and actual_len > 1:
                        weights[0] = 0  # Don't mutate first real token

                    # Normalize
                    if weights.sum() > 0:
                        weights = weights / weights.sum()

                    dist[i, start_pos:start_pos + actual_len] = weights

            return dist

        def update(self, masks):
            """Update distribution when masks change (e.g., after culling)"""
            self.dist = self._compute_distribution(masks)

    class GradientHandler(DistributionHandler):
        def _compute_distribution(self, masks):
            pass

        def update(self, estimate_bounds, masks):
            """Update distribution based on minimum gradient values"""
            if estimate_bounds is None or masks is None:
                raise ValueError("Both estimate_bounds and masks must be provided")

            n_prompts = masks.shape[0]
            seq_lengths = masks.sum(dim=1)

            # Reset distribution
            self.dist = torch.zeros_like(masks, dtype=torch.float)

            for p in range(n_prompts):
                actual_len = seq_lengths[p].item()
                if actual_len == 0:
                    continue

                # Find the first 1 in the mask (start of real tokens)
                first_real_pos = (masks[p] == 1).nonzero(as_tuple=True)[0][0].item()

                # Determine gradient start position
                if self.keep_first and actual_len > 1:
                    gradient_start_pos = first_real_pos + 1  # Skip the first real token (BOS)
                else:
                    gradient_start_pos = first_real_pos  # Start from the first real token

                # Collect gradient values for valid positions
                min_values = []
                valid_positions = []

                for pos in range(gradient_start_pos, first_real_pos + actual_len):
                    if (p, pos) in estimate_bounds:
                        min_val = estimate_bounds[(p, pos)]['min']
                        min_values.append(abs(min_val))
                        valid_positions.append(pos)

                # Normalize and set distribution
                if len(min_values) > 0:
                    min_values = torch.tensor(min_values, device=masks.device)
                    normalized = min_values / min_values.sum()

                    for i, pos in enumerate(valid_positions):
                        self.dist[p, pos] = normalized[i]

    def _create_distribution_handler(self, config):
        """Factory method for creating distribution handler"""
        if config.distribution.update_method == "gradient":
            return self.GradientHandler(config)
        else:
            return self.StaticHandler(config)
