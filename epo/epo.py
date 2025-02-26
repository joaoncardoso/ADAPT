"""
This file implements the EPO algorithm. See the `epo` function for the main entrypoint.
"""
import dataclasses
import time
import contextlib
from typing import Callable, Dict, List, Union, Tuple

import numpy as np
import torch
import torch.distributions
import torch.nn.functional as F
import transformers

from sae_lens import HookedSAETransformer

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
        self.ids = np.stack(self.ids, axis=0)
        self.target = np.stack(self.target, axis=0)
        self.xentropy = np.stack(self.xentropy, axis=0)
        self.keep = np.stack(self.keep, axis=0)
        self.runtime = np.array(self.runtime)
        self.parent_indices = np.stack(self.parent_indices, axis=0)
        self.mutation_positions = np.stack(self.mutation_positions, axis=0)
        self.sequence_lengths = np.stack(self.sequence_lengths, axis=0)
        # operation_types and buffer_sizes stay as lists since they're scalar values



@torch.no_grad()
def epo(
    cache_run: Callable,
    model: HookedSAETransformer, # becomes HookedSAETransformer
    #tokenizer: transformers.PreTrainedTokenizer, # use model.to_tokens()
    #seq_len: int = 12,
    population_size: int = 8,
    iters: int = 300,
    explore_per_pop: int = 32,
    batch_size: int = 256,
    topk: int = 512,
    #mutation_method: str = "gradient",
    x_penalty_min: float = 1.0 / 10.0,
    x_penalty_max: float = 10.0,
    ignore_fluency: bool = False,
    restart_frequency: int = 50,
    restart_xentropy: float = 2.0,
    restart_xentropy_max_mult: float = 3.0,
    seed: int = 0,
    init_prompt: str = "The quick brown fox jumps over the lazy dog",
    min_tokens: int = 8,  # Minimum allowed number of tokens
    max_tokens: int = 16,  # Maximum allowed number of tokens
    p_gcg_swap: float = 1,  # prob of an iteration being a gcg swap. 1 by default, which is equivalent to original EPO
    p_sample_insert: float = 0,  # prob of an iteration being a sampled insert
    p_sample_swap: float = 0,  # prob of an iteration being a sampled swap
    p_delete: float = 0,  # prob of an iteration being a delete
    history: History = None,
    #catch_keyboard_interrupt: bool = False,
    #callback: Union[Callable, bool] = None,
    #always_recompute_gradients: bool = False,
) -> History:
    """
    Run the EPO algorithm. See the paper for details.

    Parameters
    ----------
    cache_run
        A callable that accepts either input_ids or inputs_embeds and returns a
        dictionary containing the `target` and the logits for each token
        position.
    model
    tokenizer
    seq_len, optional
        The number of tokens in the optimized prompt, by default 16
    population_size, optional
        The population to keep at each iteration, by default 32
    iters, optional
        Number of iterations to run EPO, by default 1000
    explore_per_pop, optional
        Number of children per population member per iteration, by default 4
    batch_size, optional
        GPU batch size, by default 8
    topk, optional
        When selecting token replacements, we select the `topk` tokens by
        gradient magnitude and choose uniformly at random between those, by
        default 32.
    mutation_method, optional
        research, ignore, by default "gradient"
    x_penalty_min, optional
        The minimum cross-entropy penalty, by default 1.0/16.0
    x_penalty_max, optional
        The maximum cross-entropy penalty, by default 16.0
    restart_frequency, optional
        How often do we reset the Pareto frontier, by default 50
    restart_xentropy, optional
        When we reset the Pareto frontier, we select a population member that
        is optimal according to a cross-entropy penalty that is selected
        uniformly at random in the domain
        [restart_xentropy / restart_xentropy_max_mult,
         restart_xentropy * restart_xentropy_max_mult],
        restart_xentropy is by default 2.0
    restart_xentropy_max_mult, optional
        See the explanation for restart_xentropy, by default 3.0
    seed, optional
        Random seed used for initialization, by default 0
    initial_ids, optional
        The initial token ids to begin optimizing from. If None, the initial
        token ids will be selected randomly, by default None
    history, optional
        The history of an EPO run that we want to continue, by default None
    catch_keyboard_interrupt, optional
        Should we catch keyboard interrupts and end the EPO loop?, by default False
    callback, optional
        A function called at the beginning of each iteration, by default None
    always_recompute_gradients, optional
        If a population member is retained across an iteration, we default to
        not recomputing that population member's token gradients. If your
        cache_run stores internal state that changes, you may want to override
        this behavior and recompute gradients every iteration.

    Returns
    -------
        A History object containing the full history of the

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    ValueError
        _description_
    """
    start = time.time()
    explore_size = population_size * explore_per_pop
    device = model.device

    if seed is not None:
        torch.manual_seed(seed)

    # pad token
    pad_token_id = model.tokenizer.pad_token_id
    if pad_token_id is None:
        # Some models might not have pad_token_id set
        # In this case we should set it
        pad_token_id = model.tokenizer.eos_token_id

    if ignore_fluency:
        X = torch.zeros(population_size, device=model.device)
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
        input_ids = (model.to_tokens(init_prompt)).to(model.device)
        if not (min_tokens <= initial_ids.shape[1] <= max_tokens):
            raise ValueError(f"initial_ids must have shape (*, n) where {min_tokens} <= n <= {max_tokens}")
    else:
        history = History()
        input_ids = torch.randint(
            0, tokenizer.vocab_size, (population_size, int((min_tokens + max_tokens)/2)) # seq_len will be mid of min and max
        ).to(model.device)

    #selector = GradientSelector(model, cache_run, X, batch_size)

    op_probs = torch.tensor([p_gcg_swap, p_sample_insert, p_sample_swap, p_delete])

    # Initial state evaluation
    state = evaluate_fitness(model, cache_run, input_ids) # TODO

    #### Run the EPO loop: ####
    #if hasattr(cache_run, "setup"):
    #    cache_run.setup(input_ids)
    #state = selector.setup(input_ids)

    # Do I want to re-add the try block here?
    for i in range(iters):
        ########################################
        # 1) Report!
        ########################################
        if callback is not None:
            terminate_flag = callback(i, state, time.time() - start, history)
            if terminate_flag:
                break
            start = time.time()

        ########################################
        # 2) Birth children from parents
        # copy inputs to expand out to explore_size new candidates.
        ########################################
        operation_idx = torch.multinomial(op_probs, 1).item()
        operation = ["gcg_swap", "sample_insert", "sample_swap", "delete"][operation_idx]

        source_idx = torch.cat(
            (
                torch.arange(state.ids.shape[0], device=device).repeat(
                    explore_size // state.ids.shape[0]
                ),
                torch.arange(explore_size % state.ids.shape[0], device=device),
            )
        )
        #Do I need these?
        #assert source_idx.shape[0] == explore_size
        #assert (source_idx < state.ids.shape[0]).all()

        #new_ids = state.ids[source_idx, :].clone()

        ########################################
        # 3) Run the selector. This might be:
        #    - GCG swap
        #    - Sample insert
        #    - Sample swap
        #    - Delete
        ########################################
        #selector.mutate(state, source_idx, new_ids, topk)

        # Apply selected mutation operation
        if operation == "gcg_swap":
            candidate_ids, candidate_masks = apply_gcg_swap(
                model, cache_run, state, source_idx, topk, explore_size
            )
        elif operation == "sample_insert":
            candidate_ids, candidate_masks = apply_sample_insert(
                model, state, source_idx, explore_size
            )
        elif operation == "sample_swap":
            candidate_ids, candidate_masks = apply_sample_swap(
                model, state, source_idx, explore_size
            )
        else:  # delete
            candidate_ids, candidate_masks = apply_delete(
                state, source_idx, explore_size
            )

        ########################################
        # 4) Evaluate fitness
        ########################################
        new_state = evaluate_fitness(model, cache_run, candidate_ids, batch_size=batch_size)
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
        )

        ########################################
        # 7) Calculate gradients for the next iteration.
        ########################################

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


def gcg(
    cache_run: Callable,
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    seq_len: int = 16,
    iters: int = 1000,
    batch_size: int = 8,
    topk: int = 32,
    x_penalty_min: float = 1.0 / 16.0,
    x_penalty_max: float = 16.0,
    seed: int = 0,
    initial_ids: torch.Tensor = None,
    history: History = None,
    catch_keyboard_interrupt: bool = False,
    callback: Union[Callable, bool] = None,
    always_recompute_gradients: bool = False,
):
    """GCG is a special case of EPO where the population size is 1."""
    epo(
        cache_run,
        model,
        tokenizer,
        seq_len=seq_len,
        population_size=1,
        iters=iters,
        explore_per_pop=batch_size,
        batch_size=batch_size,
        topk=topk,
        mutation_method="gradient",
        x_penalty_min=x_penalty_min,
        x_penalty_max=x_penalty_max,
        seed=seed,
        initial_ids=initial_ids,
        history=history,
        catch_keyboard_interrupt=catch_keyboard_interrupt,
        callback=callback,
        always_recompute_gradients=always_recompute_gradients,
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
    extra: Dict[str, torch.Tensor]

    def cat(self, state2):
        return State(
            ids=torch.cat((self.ids, state2.ids), dim=0),
            attention_mask=torch.cat((self.attention_mask, state2.attention_mask), dim=0),
            target=torch.cat((self.target, state2.target), dim=0),
            xentropy=torch.cat((self.xentropy, state2.xentropy), dim=0),
            final_token=torch.cat((self.final_token, state2.final_token), dim=0),
            token_grads=cat_if_not_none(self.token_grads, state2.token_grads),
            extra={
                k: cat_if_not_none(self.extra[k], state2.extra[k]) for k in self.extra
            },
        )

    def subset(self, keep):
        return State(
            ids=self.ids[keep],
            attention_mask=self.attention_mask[keep],
            target=self.target[keep],
            xentropy=self.xentropy[keep],
            final_token=self.final_token[keep],
            token_grads=self.token_grads[keep.to("cpu")]
            if self.token_grads is not None
            else None,
            extra={k: self.extra[k][keep] for k in self.extra},
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

    def pad_to_length(self, length: int, pad_token_id: int = 0):
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
            extra=self.extra
        )


# based on https://github.com/llm-attacks/llm-attacks/blob/main/llm_attacks/gcg/gcg_attack.py
def token_grads(
    model: torch.nn.Module,
    cache_run: Callable,
    input_ids: torch.Tensor,
    x_penalty: torch.Tensor,
    batch_size: int,
):
    """
    Compute gradients with respect to one-hot encoded input tokens. This is a
    infinitesimal approximation to the token influence on the loss so it's a
    very noisy indicator of which tokens might reduce loss.
    """
    embed = model.get_input_embeddings()

    token_grads = torch.empty(
        (input_ids.shape[0], input_ids.shape[1], embed.num_embeddings),
        dtype=torch.float,
    )
    loss = torch.empty(input_ids.shape[0], device=model.device)
    xentropy = torch.empty(input_ids.shape[0], device=model.device)
    target = torch.empty(input_ids.shape[0], device=model.device)
    final_token = torch.empty(input_ids.shape[0], device=model.device, dtype=torch.long)
    extra = dict()

    with torch.enable_grad():
        model.zero_grad()

        for i in range(0, input_ids.shape[0], batch_size):
            imax = min(i + batch_size, input_ids.shape[0])

            # using a one hot matrix as input to the model gives us gradients with
            # respect to potential input tokens.
            one_hot = F.one_hot(
                input_ids[i:imax].clone(), num_classes=embed.num_embeddings
            ).to(embed.weight.dtype)
            one_hot.requires_grad = True

            cache = cache_run(inputs_embeds=torch.matmul(one_hot, embed.weight))

            logits_offset = cache["logits"][:, :-1]
            this_xentropy = (
                -(torch.log_softmax(logits_offset, dim=-1) * one_hot[:, 1:])
                .sum(dim=-1)
                .mean(dim=-1)
            )

            this_loss = -cache["target"] + this_xentropy * x_penalty[i:imax]
            this_loss.sum().backward()

            loss[i:imax] = this_loss
            target[i:imax] = cache["target"]
            xentropy[i:imax] = this_xentropy
            final_token[i:imax] = cache["logits"][:, -1, :].argmax(dim=-1)
            token_grads[i:imax] = one_hot.grad

            for k in cache:
                if k not in ["target", "logits"]:
                    e = cache[k]
                    if k not in extra:
                        extra[k] = torch.empty(
                            (input_ids.shape[0], *e.shape[1:]),
                            dtype=e.dtype,
                            device=e.device,
                        )
                    extra[k][i:imax] = e

            # important to zero out gradients here to release memory
            model.zero_grad()

    return State(input_ids, target, xentropy, final_token, token_grads, extra)


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
    
    # Create default attention mask if none provided (all 1's)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    
    # Initialize tensors for results
    target = torch.empty(input_ids.shape[0], dtype=torch.float, device=device)
    xentropy = torch.empty(input_ids.shape[0], dtype=torch.float, device=device)
    final_token = torch.empty(input_ids.shape[0], dtype=torch.long, device=device)
    extra = dict()
    
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
        
        # Store any extra outputs from cache_run
        for k in mini_batch:
            if k not in ["target", "logits"]:
                e = mini_batch[k]
                if k not in extra:
                    extra[k] = torch.empty(
                        (input_ids.shape[0], *e.shape[1:]),
                        dtype=e.dtype,
                        device=e.device,
                    )
                extra[k][i:imax] = e
    
    # Create and return a State object with all results
    return State(
        ids=input_ids,
        attention_mask=attention_mask,
        target=target,
        xentropy=xentropy,
        final_token=final_token,
        token_grads=None,
        extra=extra
    )


########################################
# Mutation operations
########################################

def apply_gcg_swap(
    model: HookedSAETransformer,
    cache_run: Callable,
    state: State,
    source_idx: torch.Tensor,
    topk: int,
    explore_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply GCG-guided token swaps to create new candidate sequences."""
    device = model.device
    pad_token_id = model.tokenizer.pad_token_id
    
    # Create copies of parent sequences and masks for mutation
    candidate_ids = state.ids[source_idx].clone()
    candidate_masks = state.attention_mask[source_idx].clone()
    
    # Get gradients through one-hot encoding
    with torch.enable_grad():
        model.zero_grad()
        
        # Convert to one-hot
        one_hot = F.one_hot(
            candidate_ids, 
            num_classes=model.W_E.shape[0]
        ).float()
        one_hot.requires_grad = True
        
        # Compute embeddings manually
        embeds = torch.matmul(one_hot, model.W_E)
        
        # Store for computation
        activation_dict = {}
        
        # Hook to inject our embeddings
        def embedding_hook(act, hook):
            activation_dict['orig_embeds'] = act.clone()
            return embeds
        
        # Run model with hook to inject embeddings
        result = model.run_with_hooks(
            input=candidate_ids,
            attention_mask=candidate_masks,
            fwd_hooks=[("hook_embed", embedding_hook)],
            reset_hooks_end=True
        )
        
        # Get target and backpropagate
        # This depends on how cache_run works in your setup
        # For now, use a placeholder approach
        target = cache_run(input_ids=candidate_ids, attention_mask=candidate_masks)['target']
        (-target).sum().backward()
        
        # Get topk alternative tokens per position based on gradients
        topk_grad = (-one_hot.grad).topk(k=topk, dim=-1)
        
        # Select random positions to mutate (only where attention mask is 1)
        # Make sure these are valid positions (not padding)
        seq_lens = candidate_masks.sum(dim=1).to(torch.long)
        pos = torch.zeros(explore_size, dtype=torch.long, device=device)
        
        for i in range(explore_size):
            # Choose a random position (not padding)
            valid_len = seq_lens[i].item()
            if valid_len > 0:
                pos[i] = torch.randint(0, valid_len, (1,), device=device)
        
        # Select random tokens from topk candidates
        token_idx = torch.randint(
            low=0,
            high=topk,
            size=(explore_size,),
            device=device
        )
        
        # Apply mutations
        for i in range(explore_size):
            candidate_ids[i, pos[i]] = topk_grad.indices[i, pos[i], token_idx[i]]

    return candidate_ids, candidate_masks

def apply_sample_insert(
    model: HookedSAETransformer,
    cache_run: Callable,
    state: State,
    source_idx: torch.Tensor,
    sample_k2: int,
    explore_size: int,
    max_tokens: int = 16,
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
    device = model.device
    pad_token_id = model.tokenizer.pad_token_id if model.tokenizer.pad_token_id is not None else model.tokenizer.eos_token_id
    
    # Get original sequences and their lengths
    original_ids = state.ids[source_idx]
    original_masks = state.attention_mask[source_idx]
    seq_lens = original_masks.sum(dim=1).to(torch.long)
    
    # Run forward pass to get token probabilities
    with torch.no_grad():
        outputs = model(input_ids=original_ids, attention_mask=original_masks)
        logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
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
        # +1 because we can insert at position 0 or at the end
        insert_pos = torch.randint(0, src_len + 1, (1,), device=device).item()
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
    cache_run: Callable,
    state: State,
    source_idx: torch.Tensor,
    sample_k2: int,
    explore_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply token swapping based on model probability to create new candidate sequences.
    
    Args:
        model: The HookedSAETransformer model
        cache_run: Function to compute target values
        state: Current state containing sequences and their attributes
        source_idx: Indices of parent sequences to mutate
        sample_k2: Number of candidate tokens to sample per position
        explore_size: Total number of candidates to generate
        
    Returns:
        Tuple of (candidate token ids, attention masks, mutation positions)
    """
    device = model.device
    
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
        outputs = model(input_ids=original_ids, attention_mask=original_masks)
        logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
        probs = torch.softmax(logits, dim=-1)  # Convert to probabilities
    
    # Process each candidate
    for i in range(explore_size):
        src_len = seq_lens[i].item()
        
        # Skip if sequence is empty or has only one token
        if src_len <= 1:
            mutation_positions[i] = 0  # No mutation possible
            continue
        
        # Choose random position to swap (not including padding)
        swap_pos = torch.randint(0, src_len, (1,), device=device).item()
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
        delete_pos = torch.randint(0, src_len, (1,), device=device).item()
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


class GradientSelector(Selector):
    uses_gradient = True

    def setup(self, input_ids: torch.Tensor):
        return token_grads(
            self.model,
            self.cache_run,
            input_ids,
            x_penalty=self.X[: input_ids.shape[0]],
            batch_size=self.batch_size,
        )

    def mutate(self, state, source_idx, input_ids, topk):
        # when just flipping, the current token gradient falls out of the
        # topk operation, so we can just use the negative new token grad
        topk_grad = (-state.token_grads).topk(k=topk, dim=-1)
        pos = torch.randint(
            low=0,
            high=input_ids.shape[1],
            size=(input_ids.shape[0],),
            device=input_ids.device,
        )
        token_idx = torch.randint(
            low=0,
            high=topk,
            size=(input_ids.shape[0],),
            device=input_ids.device,
        )
        input_ids[torch.arange(input_ids.shape[0]), pos] = topk_grad.indices.to(
            input_ids.device
        )[source_idx, pos, token_idx]


def pareto_callback(model, x_penalty_min, x_penalty_max):
    def f(i, state, last_runtime, history, final=False):
        if last_runtime is not None:
            print("runtime: {:.2f} seconds".format(last_runtime))
        print(f"\nbeginning step {i}, current pareto frontier prompts:")
        last_idx = None

        Xvs = torch.exp(
            torch.linspace(
                np.log(x_penalty_min / 10.0), np.log(x_penalty_max * 10.0), 200
            )
        ).to(model.device)
        loss = -state.target[None] + Xvs[:, None] * state.xentropy[None]
        idxs = loss.argmin(dim=1)
        for i in range(len(Xvs)):
            idx = idxs[i]
            if idx == last_idx:
                continue
            #text = tokenizer.decode(state.ids[idx])
            #last_token = tokenizer.decode(state.final_token[idx])
            text = model.to_string(state.ids[idx])
            last_token = model.to_string(state.final_token[idx])
            print(
                f"penalty={Xvs[i]:.2f} xentropy={state.xentropy[idx]:.2f} target={state.target[idx]:.2f} {repr(text + '[' + last_token + ']')}"
            )
            last_idx = idx

    return f
