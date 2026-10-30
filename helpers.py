"""
Helpers for ADAPT: the SAE feature runner, batch sizing, and the diverse beam search used for initialization.
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# ==================== #
#        RUNNER        #
# ==================== #

class SAEFeatureRunner:
    """Forward pass from input embeddings to the SAE pre-activation of a latent at the last position"""
    def __init__(self, model, sae, feature_idxs):
        self.model = model
        self.sae = sae
        self.feature_idxs = feature_idxs

    def __call__(self, inputs_embeds=None, attention_mask=None):
        layer_activations = None
        batch_size = inputs_embeds.shape[0]
        seq_length = inputs_embeds.shape[1]
        device = inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=device)

        embeds = inputs_embeds

        if hasattr(self.model, 'W_pos') and self.model.W_pos is not None:
            position_ids = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)
            pos_embeds = self.model.W_pos[position_ids]
            embeds = embeds + pos_embeds

        if hasattr(self.model, 'ln_embed') and self.model.ln_embed is not None:
            embeds = self.model.ln_embed(embeds)

        layer_name = self.sae.cfg.metadata.hook_name

        def get_layer_activation(act, hook):
            nonlocal layer_activations
            layer_activations = act
            return act

        logits = self.model.run_with_hooks(
            start_at_layer=0,
            input=embeds,
            attention_mask=attention_mask,
            fwd_hooks=[(layer_name, get_layer_activation)]
        )

        batch_indices = torch.arange(batch_size, device=device)
        last_token_activations = layer_activations[batch_indices, -1]

        sae_in = self.sae.process_sae_in(last_token_activations)
        pre_activations = torch.matmul(sae_in, self.sae.W_enc) + self.sae.b_enc
        targets = pre_activations[:, self.feature_idxs]

        del last_token_activations, embeds, attention_mask
        self.model.zero_grad()
        return {
            "target": targets,
            "logits": logits,
        }

# ==================== #
#     BATCH SIZING     #
# ==================== #

_bytes_per_token = {}  # (model, sae) -> measured memory cost of one token in a forward pass with the SAE

def is_oom(e: Exception) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e)

def _measure_bytes_per_token(model, sae, device, probe_len=16, small=4, large=16):
    """Peak memory of the evaluator's forward pass (logits, SAE hook, cross-entropy) at two batch sizes;
    the difference divided by the extra tokens is the per-token cost, independent of fixed overheads."""
    hook_name = f"{sae.cfg.metadata.hook_name}.hook_sae_acts_pre"
    peaks = []
    for b in (small, large):
        tokens = torch.randint(1, model.cfg.d_vocab, (b, probe_len), device=device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        before = torch.cuda.memory_allocated(device)
        with torch.no_grad():
            logits, cache = model.run_with_cache_with_saes(tokens, saes=[sae], use_error_term=True, names_filter=hook_name)
            ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1), reduction='none')
            half = logits.to(torch.float16)
        peaks.append(torch.cuda.max_memory_allocated(device) - before)
        del logits, cache, ce, half, tokens
    torch.cuda.empty_cache()
    return max(peaks[1] - peaks[0], 1) / ((large - small) * probe_len)

def compute_batch_size(model, sae, prompt_length, device, memory_fraction=0.8, cpu_batch_size=32):
    """Largest batch of prompts of the given length that fits in the currently free GPU memory"""
    device = torch.device(device)
    if device.type != "cuda":
        return cpu_batch_size
    key = (id(model), id(sae))
    if key not in _bytes_per_token:
        _bytes_per_token[key] = _measure_bytes_per_token(model, sae, device)
    free, _ = torch.cuda.mem_get_info(device)
    free += torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)  # cached by PyTorch, reusable
    return max(1, int(memory_fraction * free / (_bytes_per_token[key] * prompt_length)))

# ==================== #
#  DIVERSE BEAM SEARCH #
# ==================== #

def sample_initial_sequences(model, k1, start_token_id=None):
    """Sample k1 initial sequences of length 2 with diverse second tokens"""
    if start_token_id is None:
        start_token_id = model.tokenizer.bos_token_id
        if start_token_id is None:
            start_token_id = model.tokenizer.eos_token_id

    device = model.cfg.device

    # Get logits for a single BOS token (they're all identical anyway)
    single_seq = torch.tensor([[start_token_id]], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(single_seq)
        logits = logits[0, -1, :]  # [vocab_size]
        probs = F.softmax(logits, dim=-1)

    # Sample k1 distinct tokens without replacement
    next_tokens = torch.multinomial(probs, num_samples=k1, replacement=False)  # [k1]

    # Create k1 sequences: [BOS, sampled_token]
    start_tokens = torch.full((k1, 1), start_token_id, dtype=torch.long, device=device)
    next_tokens = next_tokens.unsqueeze(1)  # [k1, 1]
    sequences = torch.cat([start_tokens, next_tokens], dim=1)  # [k1, 2]

    return sequences

def _evaluate_beam(model, sae, sequences, feature_id, logit_target_indices=None):
    """Next-token logits, latent activation and (optionally) summed target-token logits at the last position"""
    total_sequences = sequences.shape[0]
    hook_name = f"{sae.cfg.metadata.hook_name}.hook_sae_acts_pre"

    final_logits = torch.empty((total_sequences, model.cfg.d_vocab), dtype=torch.float16, device=sequences.device)
    final_activation_scores = torch.empty((total_sequences,), dtype=torch.float16, device=sequences.device)
    final_logit_scores = None
    if logit_target_indices is not None:
        final_logit_scores = torch.empty((total_sequences,), dtype=torch.float16, device=sequences.device)

    def run_batch(batch):
        with torch.no_grad():
            logits, cache = model.run_with_cache_with_saes(batch, saes=[sae], use_error_term=True, names_filter=hook_name)
            activations = cache[hook_name][:, -1, feature_id]
            logit_scores = logits[:, -1, logit_target_indices].sum(dim=-1) if logit_target_indices is not None else None
            return logits[:, -1, :].to(torch.float16), activations.to(torch.float16), logit_scores

    model.zero_grad()
    torch.cuda.empty_cache()
    batch_size = compute_batch_size(model, sae, prompt_length=sequences.shape[1], device=sequences.device)

    i = 0
    while i < total_sequences:
        end_idx = min(i + batch_size, total_sequences)
        try:
            logits, activations, logit_scores = run_batch(sequences[i:end_idx])
        except Exception as e:
            if not is_oom(e) or batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
        else:
            final_logits[i:end_idx] = logits
            final_activation_scores[i:end_idx] = activations
            if logit_scores is not None:
                final_logit_scores[i:end_idx] = logit_scores.to(torch.float16)
            del logits, activations, logit_scores
            i = end_idx
        torch.cuda.empty_cache()

    return final_logits, final_activation_scores, final_logit_scores

def beam_search(model, sae, feat_id, beam_size, expansion_factor, max_length, n_groups,
                group_objectives: Optional[List[str]] = None, logit_top_k=10, init_seq_size=5000,
                top_k=128, insert_ratio=0.3, prevent_repeat=True, verbose=True) -> List[Dict]:
    """
    Diverse beam search: each of n_groups subbeams keeps its own top beam_size // n_groups sequences,
    ranked by the group's objective ("activation" of the latent, or "logit" for the latent's top output tokens).
    Candidates are sampled from the model's top-k next-token distribution, either appended or inserted.

    Returns the beam at every step, as a list of dicts with CPU tensors
    (sequences, activation_scores, logit_scores, group_ids).
    """
    device = model.cfg.device
    group_objectives = group_objectives or ["activation"] * n_groups

    # Precompute target token indices for the logit objective
    logit_target_indices = None
    if "logit" in group_objectives:
        _, logit_target_indices = torch.topk(sae.W_dec[feat_id] @ model.W_U, logit_top_k)

    history = []

    def record(sequences, activation_scores, logit_scores, group_ids):
        history.append({
            "sequences": sequences.detach().cpu().clone(),
            "activation_scores": activation_scores.detach().cpu().clone(),
            "logit_scores": logit_scores.detach().cpu().clone() if logit_scores is not None else None,
            "group_ids": group_ids.detach().cpu().clone(),
        })

    # Initial beam: top beam_size of the sampled length-2 sequences by activation,
    # with groups assigned in contiguous blocks: [0,0,0,1,1,1,2,2,2,...]
    init_sequences = sample_initial_sequences(model, init_seq_size)
    logits, activation_scores, logit_scores = _evaluate_beam(model, sae, init_sequences, feat_id, logit_target_indices)
    _, top_indices = torch.topk(activation_scores, beam_size, largest=True)
    slots_per_group = beam_size // n_groups
    group_ids = torch.clamp(torch.arange(beam_size, device=device) // slots_per_group, max=n_groups - 1)

    sequences = init_sequences[top_indices]
    logits = logits[top_indices]
    activation_scores = activation_scores[top_indices]
    logit_scores = logit_scores[top_indices] if logit_scores is not None else None
    record(sequences, activation_scores, logit_scores, group_ids)

    for iteration in range(max(0, max_length - 2)):
        # Generate candidates: n_extends appended tokens and n_inserts inserted tokens per beam
        current_length = sequences.shape[1]
        if current_length >= 3 and insert_ratio > 0:
            n_inserts = int(expansion_factor * insert_ratio)
            n_extends = expansion_factor - n_inserts
        else:
            n_inserts = 0
            n_extends = expansion_factor

        next_token_logits = logits.clone()
        if prevent_repeat:
            next_token_logits[torch.arange(beam_size, device=device), sequences[:, -1]] = float('-inf')
        next_token_probs = F.softmax(next_token_logits, dim=-1)

        candidates, parent_indices = [], []
        for i in range(beam_size):
            topk_probs, topk_indices = torch.topk(next_token_probs[i], top_k, largest=True)
            probs = topk_probs / topk_probs.sum()

            # Extensions
            if n_extends > 0:
                sampled = torch.multinomial(probs, min(n_extends, probs.size(0)), replacement=False)
                for token in topk_indices[sampled]:
                    candidates.append(torch.cat([sequences[i], token.unsqueeze(0)]))
                    parent_indices.append(i)

            # Inserts
            for _ in range(n_inserts):
                insert_pos = torch.randint(1, current_length, (1,)).item()
                sampled_token = topk_indices[torch.multinomial(probs, 1)]
                candidates.append(torch.cat([sequences[i, :insert_pos], sampled_token, sequences[i, insert_pos:]]))
                parent_indices.append(i)

        candidates = torch.stack(candidates)
        parent_indices = torch.tensor(parent_indices, device=device)
        cand_logits, cand_activation_scores, cand_logit_scores = _evaluate_beam(model, sae, candidates, feat_id, logit_target_indices)
        cand_group_ids = group_ids[parent_indices]  # inherit group from parent

        # Each group keeps its top candidates by its own objective
        selected = []
        for group_id in range(n_groups):
            group_indices = torch.where(cand_group_ids == group_id)[0]
            if len(group_indices) == 0:
                continue
            if group_objectives[group_id] == "logit" and cand_logit_scores is not None:
                group_scores = cand_logit_scores[group_indices]
            else:
                group_scores = cand_activation_scores[group_indices]
            _, top_in_group = torch.topk(group_scores, min(slots_per_group, len(group_indices)), largest=True)
            selected.extend(group_indices[top_in_group].tolist())
        selected = torch.tensor(selected, device=device)

        sequences = candidates[selected]
        logits = cand_logits[selected]
        activation_scores = cand_activation_scores[selected]
        logit_scores = cand_logit_scores[selected] if cand_logit_scores is not None else None
        group_ids = cand_group_ids[selected]
        record(sequences, activation_scores, logit_scores, group_ids)

        if verbose:
            best_idx = activation_scores.argmax()
            print(f"Beam iter {iteration}: best={activation_scores.max().item():.4f}: {model.tokenizer.decode(sequences[best_idx])!r}")

        del candidates, cand_logits, cand_activation_scores, cand_logit_scores
        torch.cuda.empty_cache()

    return history

def global_top_k(history: List[Dict], k: int = 10) -> List[Tuple[torch.Tensor, float, int]]:
    """Top-k distinct sequences by activation over the whole beam search, as (sequence, score, group_id)"""
    combined = []
    for entry in history:
        for i in range(entry["sequences"].shape[0]):
            combined.append((entry["activation_scores"][i].item(), entry["sequences"][i], entry["group_ids"][i].item()))
    combined.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate (keep first occurrence = highest score)
    seen = set()
    unique = []
    for score, seq, gid in combined:
        seq_tuple = tuple(seq.tolist())
        if seq_tuple not in seen:
            seen.add(seq_tuple)
            unique.append((seq, score, gid))
        if len(unique) >= k:
            break
    return unique

def group_bests(history: List[Dict], group_objectives: List[str]) -> Dict[int, Tuple[torch.Tensor, float, str]]:
    """Best sequence per group over the whole beam search under the group's own objective,
    as group_id -> (sequence, score, objective)"""
    bests = {}
    for entry in history:
        for i in range(entry["sequences"].shape[0]):
            gid = entry["group_ids"][i].item()
            obj = group_objectives[gid] if gid < len(group_objectives) else "activation"
            if obj == "logit" and entry["logit_scores"] is not None:
                score = entry["logit_scores"][i].item()
            else:
                score = entry["activation_scores"][i].item()
            if gid not in bests or score > bests[gid][1]:
                bests[gid] = (entry["sequences"][i].clone(), score, obj)
    return bests
