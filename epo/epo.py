"""
This file implements the EPO algorithm. See the `epo` function for the main entrypoint.
"""
import dataclasses
import time
import contextlib
from typing import Callable, Dict, List, Union, Tuple, Optional

import numpy as np
import torch
import torch.distributions
import torch.nn.functional as F
import transformers

from sae_lens import HookedSAETransformer
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

@dataclasses.dataclass
class History:
    """
    The `epo` function returns a History object that contains the full history
    of the population members at each iteration.
    """

    # The token ids for each population member at each iteration.
    ids: List = dataclasses.field(default_factory=lambda: [])
    # The sequence lengths for each population member
    lengths: List = dataclasses.field(default_factory=list)
    # The cross-entropy loss for each population member at each iteration.
    xentropy: List = dataclasses.field(default_factory=lambda: [])
    # The target objective for each population member at each iteration.
    target: List = dataclasses.field(default_factory=lambda: [])
    # The indices of the population members that were retained at each iteration.
    keep: List = dataclasses.field(default_factory=lambda: [])
    # The runtime for each iteration.
    runtime: List = dataclasses.field(default_factory=lambda: [])
    
    # New fields for mutation tracking
    operation_types: List[str] = dataclasses.field(default_factory=list)
    buffer_sizes: List[int] = dataclasses.field(default_factory=list)
    parent_indices: List[torch.Tensor] = dataclasses.field(default_factory=list)
    mutation_positions: List[torch.Tensor] = dataclasses.field(default_factory=list)

    def subset(self, slc):
        """
        Return a History object sliced along the iterations dimension.
        """
        return History(
            self.ids[slc],
            self.lengths[slc],
            self.xentropy[slc],
            self.target[slc],
            self.keep[slc],
            self.runtime[slc],
            self.operation_types[slc],
            self.buffer_sizes[slc],
            self.parent_indices[slc],
            self.mutation_positions[slc],
        )
    
    def _insert(self, new_ids, target, xentropy, keep, runtime,
                operation_type, buffer_size, parent_idx, mutation_pos, pad_token_id):
        """Insert new data into the history."""
        self.ids.append(new_ids.cpu().numpy())
        self.lengths.append((new_ids != pad_token_id).sum(dim=-1).cpu().numpy())  # Compute lengths from ids
        self.target.append(target.cpu().numpy())
        self.xentropy.append(xentropy.cpu().numpy())
        self.keep.append(keep.cpu().numpy())
        self.runtime.append(runtime)
        self.operation_types.append(operation_type)
        self.buffer_sizes.append(buffer_size)
        self.parent_indices.append(parent_idx.cpu().numpy())
        self.mutation_positions.append(mutation_pos.cpu().numpy())

    def _finalize(self):
        # Find the maximum sequence length
        max_len = max(arr.shape[1] for arr in self.ids)
        
        # Pad each array to the maximum length
        padded_ids = []
        for arr in self.ids:
            if arr.shape[1] < max_len:
                # Create padded array
                padded = np.zeros((arr.shape[0], max_len), dtype=arr.dtype)
                padded[:, :arr.shape[1]] = arr
                padded_ids.append(padded)
            else:
                padded_ids.append(arr)
        
        # Now stack the padded arrays
        self.ids = np.stack(padded_ids, axis=0)
        self.target = np.stack(self.target, axis=0)
        self.xentropy = np.stack(self.xentropy, axis=0)
        self.keep = np.stack(self.keep, axis=0)
        self.runtime = np.array(self.runtime)
        self.operation_types = np.array(self.operation_types)
        self.buffer_sizes = np.array(self.buffer_sizes)
        self.parent_indices = np.stack(self.parent_indices, axis=0)
        self.mutation_positions = np.stack(self.mutation_positions, axis=0)
        self.lengths = np.stack(self.lengths, axis=0)



@torch.no_grad()
def epo(
    cache_run: Callable,
    model: HookedSAETransformer, 
    population_size: int = 8,
    iters: int = 300,
    explore_per_pop: int = 32,
    batch_size: int = 256,
    topk: int = 512,
    x_penalty_min: float = 1.0 / 10.0,
    x_penalty_max: float = 10.0,
    ignore_fluency: bool = False,
    restart_frequency: int = 50,
    restart_xentropy: float = 2.0,
    restart_xentropy_max_mult: float = 3.0,
    seed: int = 0,
    init_prompt: Union[List[str], str] = None,
    min_tokens: int = 8,  # Minimum allowed number of tokens
    max_tokens: int = 16,  # Maximum allowed number of tokens
    sample_k2: int = 32,
    p_gcg_swap: float = 1,  # prob of an iteration being a gcg swap. 1 by default, which is equivalent to original EPO
    p_sample_insert: float = 0,  # prob of an iteration being a sampled insert
    p_sample_swap: float = 0,  # prob of an iteration being a sampled swap
    p_delete: float = 0,  # prob of an iteration being a delete
    history: History = None,
    callback: Union[Callable, bool] = None,
) -> History:
    """
    TODO: Add docstring
    """
    start = time.time()
    explore_size = population_size * explore_per_pop
    device = model.cfg.device

    if seed is not None:
        torch.manual_seed(seed)

    # pad token
    pad_token_id = model.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = model.tokenizer.eos_token_id

    if ignore_fluency:
        X = torch.zeros(population_size, device=model.cfg.device)
    else:
        X = torch.exp(torch.linspace(np.log(x_penalty_min), np.log(x_penalty_max), population_size)).to(device)

    if callback is None: # callback allows for custom behavior at each iteration
        callback = pareto_callback(
            #cache_run,
            model,
            #tokenizer,
            X.min().item(),
            X.max().item(),
        )
    elif callback is False:
        callback = lambda *x: True

    #### history and initial_ids ####
    if history is not None:
        if init_prompt is not None:
            raise ValueError("Cannot specify both history and init_prompt.")
        input_ids = history.ids[-1, history.keep[-1]]
    elif init_prompt is not None:
        history = History()
        
        # Handle case where init_prompt is a list of strings
        if isinstance(init_prompt, str):
            init_prompt = [init_prompt] * population_size
        
        # Validate that the number of prompts matches population_size
        if len(init_prompt) != population_size:
            raise ValueError(f"Number of provided prompts ({len(init_prompt)}) must match population_size ({population_size})")
        
        # Convert each prompt to token IDs
        token_ids_list = []
        for prompt in init_prompt:
            tokens = model.to_tokens(prompt).to(model.cfg.device)
            token_ids_list.append(tokens)
        
        # Check that all prompts meet length requirements
        for i, tokens in enumerate(token_ids_list):
            if not (min_tokens <= tokens.shape[1] <= max_tokens):
                raise ValueError(f"Prompt at index {i} has length {tokens.shape[1]}, but must be between {min_tokens} and {max_tokens} tokens")
        
        # Pad or truncate all sequences to the same length
        max_length = max(tokens.shape[1] for tokens in token_ids_list)
        pad_token_id = model.tokenizer.pad_token_id if model.tokenizer.pad_token_id is not None else model.tokenizer.eos_token_id
        
        input_ids = torch.full((population_size, max_length), pad_token_id, device=model.cfg.device)
        for i, tokens in enumerate(token_ids_list):
            seq_len = tokens.shape[1]
            input_ids[i, :seq_len] = tokens[0, :seq_len]
    else:
        torch.manual_seed(int(time.time()) % 100)
        # Generate random token IDs
        history = History()
        init_length = int((min_tokens + max_tokens) / 2)  # Use average of min and max for length
        input_ids = torch.randint(
            0, model.W_E.shape[0], (population_size, init_length-1)
        ).to(model.cfg.device)
        # reencode to ensure that the tokens are valid
        input_ids = model.to_tokens(model.to_string(input_ids)).to(model.cfg.device)

        if input_ids.shape[1] > max_tokens:
            input_ids = input_ids[:, :max_tokens] 
        
        # Filter out special tokens apart from the first one
        special_ids = torch.tensor(model.tokenizer.all_special_ids, device=model.cfg.device)
        for i in range(population_size):
            for j in range(1, init_length-1):
                while input_ids[i, j] in special_ids:
                    input_ids[i, j] = torch.randint(0, model.W_E.shape[0], (1,), device=model.cfg.device)


    op_probs = torch.tensor([p_gcg_swap, p_sample_insert, p_sample_swap, p_delete])

    # Initial state evaluation
    state = evaluate_fitness(model, cache_run, input_ids, pad_token_id=pad_token_id)

    #### Run the EPO loop: ####

    for i in range(iters):

        if i == 15:
            None
        ########################################
        # 1) Report!
        ########################################
        if callback is not None:
            terminate_flag = callback(i, state, time.time() - start, history)
            if terminate_flag:
                break
            start = time.time()

        # select operations
        operation_idx = torch.multinomial(op_probs, 1).item()
        operation = ["gcg_swap", "sample_insert", "sample_swap", "delete"][operation_idx]

        # compute gradients if needed
        if operation == "gcg_swap":
            state = compute_gradients(model, cache_run, state.ids, state.attention_mask, X)
            if state.token_grads[0] is None:
                raise ValueError("Token gradients were not computed wtfff????!?.")


        ########################################
        # 2) Birth children from parents
        # copy inputs to expand out to explore_size new candidates.
        ########################################

        source_idx = torch.cat(
            (
                torch.arange(state.ids.shape[0], device=device).repeat(
                    explore_size // state.ids.shape[0]
                ),
                torch.arange(explore_size % state.ids.shape[0], device=device),
            )
        )

        ########################################
        # 3) Run the selector. This might be:
        #    - GCG swap
        #    - Sample insert
        #    - Sample swap
        #    - Delete
        ########################################

        # Apply selected mutation operation
        if operation == "gcg_swap":
            candidate_ids, candidate_masks, mutation_positions  = apply_gcg_swap(
                state, source_idx, topk, explore_size, model.tokenizer.all_special_ids, model.cfg
            )
        elif operation == "sample_insert":
            candidate_ids, candidate_masks, mutation_positions = apply_sample_insert(
                model, state, source_idx, explore_size, sample_k2, max_tokens, model.tokenizer.all_special_ids
            )
        elif operation == "sample_swap":
            candidate_ids, candidate_masks, mutation_positions = apply_sample_swap(
                model, state, source_idx, explore_size, sample_k2, model.tokenizer.all_special_ids
            )
        else:  # delete
            candidate_ids, candidate_masks, mutation_positions = apply_delete(
                state, source_idx, explore_size, min_tokens
            )

        ########################################
        # 4) Evaluate fitness
        ########################################
        new_state = evaluate_fitness(model, cache_run, candidate_ids, candidate_masks, batch_size=batch_size, pad_token_id=pad_token_id)
        print(f"state dimensions: {state.ids.shape}, new_state dimensions: {new_state.ids.shape}")
        all_state = state.cat(new_state)

        # note that all_loss is a matrix with a row for each population
        # member because each population member slot uses a different
        # xentropy penalty.
        all_loss = (
            -all_state.target[None, :] + X[:, None] * all_state.xentropy[None, :]
        )
        keep = (-all_loss).argmax(dim=1).to(torch.int)

        ########################################
        # 5) Handle restarts
        ########################################

        if i > 0 and i % restart_frequency == 0:
            min_mult = 1.0 / restart_xentropy_max_mult
            max_mult = restart_xentropy_max_mult
            mult = min_mult + (max_mult - min_mult) * torch.rand(1).item()
            restart_X = restart_xentropy * mult
            restart_loss = -all_state.target + restart_xentropy * all_state.xentropy
            print(f"restarting with xentropy penalty of {restart_X:.2f}")
            keep[:] = restart_loss.argmin()

        ########################################
        # 6) Update history
        ########################################
        history._insert(
            all_state.ids,
            all_state.target,
            all_state.xentropy,
            keep,
            time.time() - start,
            operation,
            len(keep),
            source_idx,
            mutation_positions,  # Need to track this in mutation operations
            pad_token_id
        )

        state = all_state.subset(keep)

        """ # do i need this?
        if i != iters - 1:
            if operation == "gcg_swap" and not recompute_gradients:
                # Only recompute gradients for new sequences
                survived = keep[keep < state.ids.shape[0]]
                new = keep[keep >= state.ids.shape[0]]
                if new.shape[0] > 0:
                    state_new = compute_gradients(all_state.ids[new])  # New helper function
                if survived.shape[0] > 0:
                    state_survived = state.subset(survived)
                    state = state_survived.cat(state_new) if new.shape[0] > 0 else state_survived
                else:
                    state = state_new
            else:
                state = all_state.subset(keep)
        """

    terminate_flag = callback(i, state, time.time() - start, history, final=True)

    history._finalize()
    return history


@dataclasses.dataclass
class ParetoFrontier:
    # the range of cross-entropy penalties used
    Xvs: np.ndarray
    # the target and xentropy values for each penalty level
    full_target: np.ndarray
    full_xentropy: np.ndarray
    # the unique indices in full_target/full_xentropy that make up the pareto frontier.
    unique: np.ndarray
    # the target and xentropy values for the unique entries
    target: np.ndarray
    xentropy: np.ndarray
    # the token ids for each unique point on the frontier.
    ids: np.ndarray
    # the detokenized text for each unique point on the frontier.
    text: List[str]


def build_pareto_frontier(tokenizer, histories, Xvs=None):
    """
    Construct a pareto frontier from the history of several EPO runs. We allow
    multiple histories to be passed so that we can construct the Pareto
    frontier across several different runs of EPO with different random
    initializations.

    Parameters
    ----------
    tokenizer
    histories
        A list of History objects returned by the EPO algorithm. We allow
        multiple independent histories to be combined
    Xvs, optional
        The range of cross-entropy penalties to use.
        By default Xvs = 1.0 / np.linspace(0, 50, 1000)[1:]

    Returns
    -------
        A ParetoFrontier object.
    """

    if Xvs is None:
        Xvs = 1.0 / np.linspace(0, 50, 1000)[1:]

    if not isinstance(histories, list):
        histories = [histories]
    x = []
    t = []
    ids = []
    for h in histories:
        x.append(h.xentropy.flatten())
        t.append(h.target.flatten())
        ids.append(h.ids.reshape((-1, h.ids.shape[-1])))

    history_x = np.concatenate(x)
    history_t = np.concatenate(t)
    history_ids = np.concatenate(ids, axis=0)
    pareto_t = np.empty(Xvs.shape[0])
    pareto_x = np.empty(Xvs.shape[0])
    pareto_idxs = []
    for i, Xv in enumerate(Xvs):
        loss = -history_t + Xv * history_x
        idx = loss.argmin()
        pareto_idxs.append(idx)
        pareto_t[i] = history_t[idx]
        pareto_x[i] = history_x[idx]
    pareto_unique = np.unique(pareto_idxs, return_index=True)[1]
    pareto_ids = [history_ids[pareto_idxs[i]] for i in pareto_unique]
    pareto_text = [tokenizer.decode(ids) for ids in pareto_ids]
    return ParetoFrontier(
        np.array(Xvs),
        pareto_t,
        pareto_x,
        pareto_unique,
        pareto_t[pareto_unique],
        pareto_x[pareto_unique],
        pareto_ids,
        pareto_text,
    )

########################################
# Private implementation details below here.
########################################


def cat_if_not_none(a, b):
    if a is None or b is None:
        return None
    else:
        return torch.cat((a, b), dim=0)


@dataclasses.dataclass
class State:
    ids: torch.Tensor
    # The attention mask for each sequence (1 for real tokens, 0 for padding)
    attention_mask: torch.Tensor  # shape: (batch_size, max_seq_len)
    target: torch.Tensor
    xentropy: torch.Tensor
    final_token: torch.Tensor
    token_grads: torch.Tensor

    def cat(self, state2):
        """
        Concatenate two states, handling different sequence lengths by padding.
        """
        
        # Determine the maximum sequence length across both states
        max_seq_len = max(self.ids.shape[1], state2.ids.shape[1])
        pad_token_id = 0  # Assuming 0 is the pad token, adjust if needed
        
        # Function to pad a tensor to max_seq_len
        def pad_tensor(tensor, pad_value=0):
            if tensor.shape[1] == max_seq_len:
                return tensor
                
            # Calculate padding size
            pad_size = max_seq_len - tensor.shape[1]
            
            # Handle tensors with different dimensions
            if len(tensor.shape) == 2:  # [batch_size, seq_len]
                # Pad only the sequence dimension (dim=1)
                return torch.nn.functional.pad(tensor, (0, pad_size), "constant", pad_value)
            elif len(tensor.shape) == 3:  # [batch_size, seq_len, hidden]
                # Pad only the sequence dimension (dim=1), not the hidden dimension
                return torch.nn.functional.pad(tensor, (0, 0, 0, pad_size), "constant", pad_value)
            else:
                raise ValueError(f"Unsupported tensor shape: {tensor.shape}")
        
        # Pad ids and attention mask for both states
        ids1_padded = pad_tensor(self.ids, pad_token_id)
        ids2_padded = pad_tensor(state2.ids, pad_token_id)
        mask1_padded = pad_tensor(self.attention_mask, 0)
        mask2_padded = pad_tensor(state2.attention_mask, 0)
        
        # Handle token_grads specially as they're a list of tensors
        token_grads = None
        if self.token_grads is not None and state2.token_grads is not None:
            # Assuming token_grads is a list of tensors or None values
            token_grads = self.token_grads + state2.token_grads
        
        return State(
            ids=torch.cat((ids1_padded, ids2_padded), dim=0),
            attention_mask=torch.cat((mask1_padded, mask2_padded), dim=0),
            target=torch.cat((self.target, state2.target), dim=0),
            xentropy=torch.cat((self.xentropy, state2.xentropy), dim=0),
            final_token=torch.cat((self.final_token, state2.final_token), dim=0),
            token_grads=token_grads,
        )

    def subset(self, keep):
        keep_indices = keep.tolist()
        return State(
            ids=self.ids[keep],
            attention_mask=self.attention_mask[keep],
            target=self.target[keep],
            xentropy=self.xentropy[keep],
            final_token=self.final_token[keep],
            token_grads=[self.token_grads[i] for i in keep_indices]
            if self.token_grads is not None
            else None,
        )
    
    @property
    def lengths(self):
        """Get the actual length of each sequence"""
        return self.attention_mask.sum(dim=-1)

    @classmethod
    def create(cls, ids: torch.Tensor, pad_token_id: int, **kwargs):
        """Create a new state from token ids, automatically generating the attention mask"""
        # Create attention mask (1s for real tokens, 0s for padding)
        attention_mask = (ids != pad_token_id).to(torch.long)
        return cls(ids=ids, attention_mask=attention_mask, **kwargs)

    def pad_to_length(self, length: int, pad_token_id: int = 0): # not used lol
        """Pad all sequences in the state to the specified length"""
        if length <= self.ids.shape[1]:
            return self
            
        # Calculate padding amounts
        pad_length = length - self.ids.shape[1]
        
        # Pad ids
        new_ids = F.pad(
            self.ids, 
            (0, pad_length),
            mode='constant',
            value=pad_token_id
        )
        
        # Pad attention mask with zeros
        new_attention_mask = F.pad(
            self.attention_mask,
            (0, pad_length),
            mode='constant',
            value=0
        )
        
        # Pad token gradients if they exist
        new_token_grads = None
        if self.token_grads is not None:
            new_token_grads = F.pad(
                self.token_grads,
                (0, 0, 0, pad_length),
                mode='constant',
                value=0
            )
            
        return State(
            ids=new_ids,
            attention_mask=new_attention_mask,
            target=self.target,
            xentropy=self.xentropy,
            final_token=self.final_token,
            token_grads=new_token_grads,
        )


def calc_xentropy(logits, input_ids):
    logits_offset = logits[:, :-1]
    return (
        torch.nn.CrossEntropyLoss(reduction="none")(
            logits_offset.reshape(-1, logits_offset.shape[-1]),
            input_ids[:, 1:].reshape(-1),
        )
        .view(*logits_offset.shape[:2])
        .mean(dim=-1)
    )


def evaluate_fitness(
    model: torch.nn.Module,
    cache_run: Callable,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor = None,
    batch_size: int = 32,
    pad_token_id: int = 0,
):
    """
    Evaluate the fitness of candidate sequences, handling variable-length sequences.
    
    Args:
        model: The model to evaluate with
        cache_run: Function to compute target values
        input_ids: Candidate token ids [batch_size, seq_len]
        attention_mask: Attention masks (1 for tokens, 0 for padding) [batch_size, seq_len]
        batch_size: Batch size for processing
        
    Returns:
        State object containing evaluation results
    """
    device = input_ids.device
    
    if attention_mask is None:
        attention_mask = (input_ids != pad_token_id).long().to(device)
    # Initialize tensors for results
    target = torch.empty(input_ids.shape[0], dtype=torch.float, device=device)
    xentropy = torch.empty(input_ids.shape[0], dtype=torch.float, device=device)
    final_token = torch.empty(input_ids.shape[0], dtype=torch.long, device=device)
    
    # Process in batches
    for i in range(0, input_ids.shape[0], batch_size):
        imax = min(i + batch_size, input_ids.shape[0])
        batch_ids = input_ids[i:imax]
        batch_mask = attention_mask[i:imax]
        
        # Run the model on this batch
        mini_batch = cache_run(input_ids=batch_ids, attention_mask=batch_mask)
        
        # Store target values
        target[i:imax] = mini_batch["target"]
        
        # Calculate cross-entropy only on real tokens (not padding)
        logits = mini_batch["logits"][:, :-1]  # Remove the last position
        shift_ids = batch_ids[:, 1:]  # Shift right to align with logits
        shift_mask = batch_mask[:, 1:]  # Also shift the mask
        
        # Calculate loss with ignore_index for padding
        loss = torch.nn.CrossEntropyLoss(reduction='none')(
            logits.reshape(-1, logits.shape[-1]),
            shift_ids.reshape(-1)
        ).reshape(logits.shape[0], -1)
        
        # Mask out padding tokens and average over real tokens
        masked_loss = loss * shift_mask
        seq_lengths = shift_mask.sum(dim=1).clamp(min=1)  # Avoid division by zero
        xentropy[i:imax] = masked_loss.sum(dim=1) / seq_lengths
        
        # Get the predicted next token
        final_token[i:imax] = logits[:, -1, :].argmax(dim=-1)
    
    # Create and return a State object with all results
    return State(
        ids=input_ids,
        attention_mask=attention_mask,
        target=target,
        xentropy=xentropy,
        final_token=final_token,
        token_grads= [None] * input_ids.shape[0],
    )


########################################
# Mutation operations
########################################

def compute_gradients(
    model: HookedSAETransformer,
    cache_run: Callable,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    x_penalty: torch.Tensor,
    token_grads: List[Optional[torch.Tensor]] = None,
    batch_size: int = 32,
) -> State:
    """
    Compute gradients with respect to one-hot encoded input tokens.
    
    Args:
        model: The model to compute gradients for
        cache_run: Function that computes target values
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask (1 for tokens, 0 for padding)
        x_penalty: Cross-entropy penalty for each sequence
        token_grads: Current token gradients list (can contain None entries)
        batch_size: Batch size for processing
        
    Returns:
        State object containing gradients and other information
    """
    embed = model.W_E
    vocab_size = embed.shape[0]
    device = embed.device

    # Initialize containers for results
    batch_size_total = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    if token_grads is None:
        token_grads = [None] * batch_size_total

    # we want to compute grads only for the sequences that don't already have them
    grad_need_indices = [i for i in range(batch_size_total) if token_grads[i] is None]
    if not grad_need_indices:
        return State(
            ids=input_ids,
            attention_mask=attention_mask,
            target=torch.zeros(batch_size_total, dtype=torch.float, device=device),
            xentropy=torch.zeros(batch_size_total, dtype=torch.float, device=device),
            final_token=torch.zeros(batch_size_total, dtype=torch.long, device=device),
            token_grads=token_grads,
        )
    
    target = torch.zeros(batch_size_total, device=device)
    xentropy = torch.zeros(batch_size_total, device=device)
    final_token = torch.zeros(batch_size_total, device=device, dtype=torch.long)

    with torch.enable_grad():
        model.zero_grad()
        
        for i in range(0, len(grad_need_indices), batch_size):
            batch_indices = grad_need_indices[i:i+batch_size]
            
            # Get batch data
            batch_input_ids = input_ids[batch_indices]
            batch_attention_mask = attention_mask[batch_indices]
            batch_x_penalty = x_penalty[batch_indices]
            
            # One-hot encode with gradients
            one_hot = torch.nn.functional.one_hot(
                batch_input_ids, num_classes=vocab_size
            ).to(embed.dtype).to(device)
            one_hot.requires_grad = True
            
            # Forward pass with cache_run
            embeds = torch.matmul(one_hot, embed)
            result = cache_run(
                inputs_embeds=embeds,
                attention_mask=batch_attention_mask
            )
            
            # Calculate cross-entropy (masked for variable-length sequences)
            logits = result["logits"]
            this_target = result["target"]
            
            logits_offset = logits[:, :-1]
            targets = batch_input_ids[:, 1:]
            mask = batch_attention_mask[:, 1:]
            
            # Compute loss with masking
            loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            token_losses = loss_fn(
                logits_offset.reshape(-1, logits_offset.shape[-1]),
                targets.reshape(-1)
            ).reshape(logits_offset.shape[0], -1)
            
            # Apply mask and calculate mean over actual tokens
            masked_loss = token_losses * mask
            seq_lengths = mask.sum(dim=1).clamp(min=1)  # Avoid div by zero
            this_xentropy = masked_loss.sum(dim=1) / seq_lengths
            
            # Get target values and compute loss with penalty
            this_loss = -this_target + this_xentropy * batch_x_penalty
            
            # Backpropagate
            this_loss.sum().backward()

            target[batch_indices] = this_target
            xentropy[batch_indices] = this_xentropy
            final_token[batch_indices] = logits[:, -1, :].argmax(dim=-1)

            for j, idx in enumerate(batch_indices):
                # Store the computed gradients in the token_grads list
                token_grads[idx] = one_hot.grad[j].clone().detach()

            
            # Clear gradients for next batch
            model.zero_grad()

        return State(
            ids=input_ids,
            attention_mask=attention_mask,
            target=target,
            xentropy=xentropy,
            final_token=final_token,
            token_grads=token_grads,
        )


def apply_gcg_swap(
    state: State,
    source_idx: torch.Tensor,
    topk: int,
    explore_size: int,
    special_ids: Optional[torch.Tensor] = None,
    cfg: HookedTransformerConfig = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply GCG swap using pre-computed gradients from the state.
    
    Args:
        state: State object containing pre-computed token gradients
        source_idx: Indices of parent sequences to use for creating children
        topk: Number of top tokens to consider for each position
        explore_size: Total number of child candidates to generate
        
    Returns:
        Tuple of (candidate token ids, attention masks, mutation positions)
    """
    device = state.ids.device
    vocab_size = cfg.d_vocab
    #vocab_size = state.token_grads[0].shape[-1] # assuming no entries of token_grads are None

    
    # Create copies of parent sequences and masks for mutation
    candidate_ids = state.ids[source_idx].clone()
    candidate_masks = state.attention_mask[source_idx].clone()
    
    # Get sequence lengths (excluding padding)
    seq_lens = candidate_masks.sum(dim=1).to(torch.long)
    
    # For tracking which positions were modified
    mutation_positions = torch.zeros(explore_size, dtype=torch.long, device=device)
    
    # Select random positions and random tokens from top-k
    for i in range(explore_size):
        parent_idx = source_idx[i].item()
        parent_grads = state.token_grads[parent_idx]  # Get the parent's gradients

        # Only select positions with real tokens (not padding)
        valid_len = seq_lens[i].item()
        if valid_len > 0:
            # Choose random position within sequence length
            pos = torch.randint(0, valid_len, (1,), device=device).item()
            mutation_positions[i] = pos
            

            # Get top-k tokens for this position based on gradients
            pos_grads = -parent_grads[pos]  # Negative because we want to decrease loss

            # If eos_token_id is provided, mask it out to prevent selection
            if special_ids is not None:
                pos_grads[special_ids] = -float("inf")

            topk_values, topk_indices = pos_grads.topk(k=min(topk, vocab_size))
            
            # Choose random token from top-k
            token_idx = torch.randint(0, len(topk_indices), (1,), device=device).item()

            # Apply the mutation
            candidate_ids[i, pos] = topk_indices[token_idx]
    
    return candidate_ids, candidate_masks, mutation_positions


def apply_sample_insert(
    model: HookedSAETransformer,
    state: State,
    source_idx: torch.Tensor,
    explore_size: int,
    sample_k2: int = 32,
    max_tokens: int = 16,
    special_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply token insertion based on model probability to create new candidate sequences.
    
    Args:
        model: The HookedSAETransformer model
        cache_run: Function to compute target values
        state: Current state containing sequences and their attributes
        source_idx: Indices of parent sequences to mutate
        sample_k2: Number of candidate tokens to sample per position
        explore_size: Total number of candidates to generate
        max_tokens: Maximum allowed sequence length
        
    Returns:
        Tuple of (candidate token ids, attention masks)
    """
    device = model.cfg.device
    pad_token_id = model.tokenizer.pad_token_id if model.tokenizer.pad_token_id is not None else model.tokenizer.eos_token_id
    
    # Get original sequences and their lengths
    original_ids = state.ids[source_idx]
    original_masks = state.attention_mask[source_idx]
    seq_lens = original_masks.sum(dim=1).to(torch.long)
    
    # Run forward pass to get token probabilities
    with torch.no_grad():
        logits = model(input=original_ids, attention_mask=original_masks)
        # set logits of special tokens to -inf
        if special_ids is not None:
            logits[:, :, special_ids] = float("-inf")

        probs = torch.softmax(logits, dim=-1)  # Convert to probabilities
    
    # Initialize containers for new sequences
    # We'll need to pad by 1 since we're inserting
    max_orig_len = original_ids.shape[1]
    new_seq_len = min(max_orig_len + 1, max_tokens)
    candidate_ids = torch.full((explore_size, new_seq_len), pad_token_id, device=device)
    candidate_masks = torch.zeros((explore_size, new_seq_len), device=device, dtype=torch.long)
    
    # For tracking which positions were modified
    mutation_positions = torch.zeros(explore_size, device=device, dtype=torch.long)
    
    # Process each candidate
    for i in range(explore_size):
        src_idx = source_idx[i]  # Get the parent index
        src_len = seq_lens[i].item()  # Length of parent sequence
        
        # Skip if sequence is already at max length
        if src_len >= max_tokens:
            # Just copy the original sequence (truncated to max_tokens)
            candidate_ids[i, :] = original_ids[i, :new_seq_len]
            candidate_masks[i, :] = original_masks[i, :new_seq_len]
            mutation_positions[i] = 0  # No mutation
            continue
            
        # Choose random position to insert (can be at the end)
        # +1 because we can insert at position 1 or at the end
        insert_pos = torch.randint(1, src_len + 1, (1,), device=device).item()
        mutation_positions[i] = insert_pos
        
        # Sample tokens based on position's next token distribution
        # If inserting at the end, use the last position's distribution
        prob_pos = min(insert_pos, src_len - 1) if src_len > 0 else 0
        position_probs = probs[i, prob_pos]
        
        # Sample k2 tokens without replacement
        sampled_tokens = torch.multinomial(position_probs, num_samples=min(sample_k2, position_probs.shape[0]), replacement=False)
        
        # Choose one random token from the samples
        insert_token = sampled_tokens[torch.randint(0, len(sampled_tokens), (1,))].item()

        # Create the new sequence by inserting the token
        if insert_pos == 0:
            # Insert at the beginning
            candidate_ids[i, 0] = insert_token
            candidate_ids[i, 1:src_len+1] = original_ids[i, :src_len]
            candidate_masks[i, :src_len+1] = 1
        elif insert_pos == src_len:
            # Insert at the end
            candidate_ids[i, :src_len] = original_ids[i, :src_len]
            candidate_ids[i, src_len] = insert_token
            candidate_masks[i, :src_len+1] = 1
        else:
            # Insert in the middle
            candidate_ids[i, :insert_pos] = original_ids[i, :insert_pos]
            candidate_ids[i, insert_pos] = insert_token
            candidate_ids[i, insert_pos+1:src_len+1] = original_ids[i, insert_pos:src_len]
            candidate_masks[i, :src_len+1] = 1
    
    return candidate_ids, candidate_masks, mutation_positions

def apply_sample_swap(
    model: HookedSAETransformer,
    state: State,
    source_idx: torch.Tensor,
    explore_size: int,
    sample_k2: int,
    special_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply token swapping based on model probability to create new candidate sequences.
    
    Args:
        model: The HookedSAETransformer model
        cache_run: Function to compute target values
        state: Current state containing sequences and their attributes
        source_idx: Indices of parent sequences to mutate
        explore_size: Total number of candidates to generate
        sample_k2: Number of candidate tokens to sample per position
        
    Returns:
        Tuple of (candidate token ids, attention masks, mutation positions)
    """
    device = model.cfg.device
    
    # Get original sequences and their masks
    original_ids = state.ids[source_idx]
    original_masks = state.attention_mask[source_idx]
    seq_lens = original_masks.sum(dim=1).to(torch.long)
    
    # Create copies for mutation
    candidate_ids = original_ids.clone()
    candidate_masks = original_masks.clone()
    
    # For tracking which positions were modified
    mutation_positions = torch.zeros(explore_size, device=device, dtype=torch.long)
    
    # Run forward pass to get token probabilities
    with torch.no_grad():
        logits = model(input=original_ids, attention_mask=original_masks)
        if special_ids is not None:
            logits[:, :, special_ids] = float("-inf")
        probs = torch.softmax(logits, dim=-1)  # Convert to probabilities
    
    # Process each candidate
    for i in range(explore_size):
        src_len = seq_lens[i].item()
        
        # Skip if sequence is empty or has only one token
        if src_len <= 1:
            mutation_positions[i] = 0  # No mutation possible
            continue
        
        # Choose random position to swap (not including padding)
        swap_pos = torch.randint(1, src_len, (1,), device=device).item()
        mutation_positions[i] = swap_pos
        
        # Sample tokens based on this position's distribution
        position_probs = probs[i, swap_pos]
        
        # Sample k2 tokens without replacement
        sampled_tokens = torch.multinomial(
            position_probs, 
            num_samples=min(sample_k2, position_probs.shape[0]), 
            replacement=False
        )
        
        # Choose one random token from the samples
        swap_token = sampled_tokens[torch.randint(0, len(sampled_tokens), (1,))].item()
        
        # Apply the swap
        candidate_ids[i, swap_pos] = swap_token
    
    return candidate_ids, candidate_masks, mutation_positions

def apply_delete(
    state: State,
    source_idx: torch.Tensor,
    explore_size: int,
    min_tokens: int = 8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply token deletion to create new candidate sequences.
    
    Args:
        state: Current state containing sequences and their attributes
        source_idx: Indices of parent sequences to mutate
        explore_size: Total number of candidates to generate
        min_tokens: Minimum allowed sequence length
        
    Returns:
        Tuple of (candidate token ids, attention masks, mutation positions)
    """
    device = state.ids.device
    pad_token_id = state.ids[0, -1].item()  # Assuming padding is at the end
    
    # Get original sequences and their lengths
    original_ids = state.ids[source_idx]
    original_masks = state.attention_mask[source_idx]
    seq_lens = original_masks.sum(dim=1).to(torch.long)
    
    # Initialize containers for new sequences (1 token shorter)
    max_orig_len = original_ids.shape[1]
    new_seq_len = max_orig_len - 1  # We're deleting one token
    candidate_ids = torch.full((explore_size, new_seq_len), pad_token_id, device=device)
    candidate_masks = torch.zeros((explore_size, new_seq_len), device=device, dtype=torch.long)
    
    # For tracking which positions were modified
    mutation_positions = torch.zeros(explore_size, device=device, dtype=torch.long)
    
    # Process each candidate
    for i in range(explore_size):
        src_len = seq_lens[i].item()
        
        # Skip if sequence would become too short after deletion
        if src_len <= min_tokens:
            # Copy the original but truncate to new_seq_len
            candidate_ids[i, :] = original_ids[i, :new_seq_len]
            candidate_masks[i, :] = original_masks[i, :new_seq_len]
            mutation_positions[i] = -1  # Indicate no deletion occurred
            continue
        
        # Choose random position to delete (only from non-padding tokens)
        delete_pos = torch.randint(1, src_len, (1,), device=device).item()
        mutation_positions[i] = delete_pos
        
        # Create the new sequence by removing the token
        if delete_pos == 0:
            # Delete at the beginning
            candidate_ids[i, :src_len-1] = original_ids[i, 1:src_len]
            candidate_masks[i, :src_len-1] = 1
        elif delete_pos == src_len - 1:
            # Delete at the end
            candidate_ids[i, :src_len-1] = original_ids[i, :src_len-1]
            candidate_masks[i, :src_len-1] = 1
        else:
            # Delete in the middle
            candidate_ids[i, :delete_pos] = original_ids[i, :delete_pos]
            candidate_ids[i, delete_pos:src_len-1] = original_ids[i, delete_pos+1:src_len]
            candidate_masks[i, :src_len-1] = 1
    
    return candidate_ids, candidate_masks, mutation_positions



### Todo remove selectors
class Selector:
    def __init__(
        self,
        model: torch.nn.Module,
        cache_run: Callable,
        X: torch.Tensor,
        batch_size: int,
    ):
        self.model = model
        self.cache_run = cache_run
        self.X = X
        self.batch_size = batch_size


def pareto_callback(model, x_penalty_min, x_penalty_max):
    def f(i, state, last_runtime, history, final=False):
        if last_runtime is not None:
            print("runtime: {:.2f} seconds".format(last_runtime))

        print(f"\nbeginning step {i}, State has shape {state.ids.shape}")
        print("Current pareto frontier prompts:")
        last_idx = None

        Xvs = torch.exp(
            torch.linspace(
                np.log(x_penalty_min / 10.0), np.log(x_penalty_max * 10.0), 200
            )
        ).to(model.cfg.device)
        loss = -state.target[None] + Xvs[:, None] * state.xentropy[None]
        idxs = loss.argmin(dim=1)
        for i in range(len(Xvs)):
            idx = idxs[i]
            if idx == last_idx:
                continue
             # Use attention mask to get only real tokens
            if hasattr(state, 'attention_mask'):
                # Get sequence length from attention mask
                seq_len = state.attention_mask[idx].sum().item()
                # Get only the non-padding tokens
                tokens = state.ids[idx, :seq_len]
                text = model.to_string(tokens)
            else:
                # Fallback if no attention mask
                text = model.to_string(state.ids[idx])
                
            last_token = model.to_string(state.final_token[idx])
            
            print(
                f"penalty={Xvs[i]:.2f} xentropy={state.xentropy[idx]:.2f} "
                f"target={state.target[idx]:.2f} {repr(text + '[' + last_token + ']')}"
            )
            
            last_idx = idx
            
    return f
