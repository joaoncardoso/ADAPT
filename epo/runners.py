from typing import Optional, Tuple, List, Callable
import contextlib
import torch
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb

def does_retokenize(model, input_ids):
    good = torch.empty(input_ids.shape[0], dtype=bool).to(model.cfg.device)
    input_strs = model.to_string(input_ids)
    for i, s in enumerate(input_strs):
        model.tokenizer.add_bos_token = False
        retokenized = model.to_tokens(s)
        if retokenized.shape[1] != input_ids.shape[1]:
            good[i] = False
        else:
            good[i] = (retokenized[0] == input_ids[i]).all()
        if not good[i]:
            print(f"bad input {i}: {s}")
    model.tokenizer.add_bos_token = True
    return good


@contextlib.contextmanager
def add_fwd_hooks(module_hooks: List[Tuple[torch.nn.Module, Callable]]):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for mod, hk in module_hooks:
            handles.append(mod.register_forward_hook(hk))
        yield
    finally:
        for h in handles:
            h.remove()


def sae_feature_runner(model, sae, feature_idx, check_retokenization=True):
    """
    Create a runner function for optimizing prompts to maximize SAE feature activation.
   
    Args:
        model: The HookedSAETransformer model
        sae: The sparse autoencoder
        feature_idx: Index of the feature to target
        use_post: If True, use post-activation values, otherwise use pre-activation
   
    Returns:
        A function that computes gradients for token substitution with batch support
    """
    def run(input_ids=None, inputs_embeds=None, attention_mask=None):
        # Storage for activations
        layer_activations = None
        good = None
        # Get batch size
        if input_ids is not None:
            batch_size = input_ids.shape[0]
            seq_length = input_ids.shape[1]
            device = input_ids.device
            # Check if input_ids are correctly tokenized
            if check_retokenization:
                good = does_retokenize(model, input_ids)
        else:
            batch_size = inputs_embeds.shape[0]
            seq_length = inputs_embeds.shape[1]
            device = inputs_embeds.device
            
        # Create default attention mask if none provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=device)
        
        # For input_ids path: create one-hot encoding with gradients
        if input_ids is not None:
            one_hot = torch.nn.functional.one_hot(input_ids, num_classes=model.W_E.shape[0]).float()
            one_hot = one_hot.to(model.W_E.dtype).to(model.W_E.device)
            one_hot.requires_grad_(True)
            
            # Create embeddings
            embeds = torch.matmul(one_hot, model.W_E)
            
            # Add positional embeddings if available
            if hasattr(model, 'W_pos') and model.W_pos is not None:
                position_ids = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)
                pos_embeds = model.W_pos[position_ids]
                embeds = embeds + pos_embeds
                
            # Apply layer normalization if available
            if hasattr(model, 'ln_embed') and model.ln_embed is not None:
                embeds = model.ln_embed(embeds)
        else:
            # For inputs_embeds path: use the provided embeddings
            embeds = inputs_embeds
            one_hot = None
       
        # Get the layer from the SAE hook name
        layer_name = sae.cfg.hook_name
       
        # Hook to capture the layer activations
        def get_layer_activation(act, hook):
            nonlocal layer_activations
            layer_activations = act
            return act
       
        # Run model with hooks
        logits = model.run_with_hooks(
            start_at_layer=0,
            input=embeds,
            attention_mask=attention_mask,
            fwd_hooks=[(layer_name, get_layer_activation)]
        )
        
        # Calculate sequence lengths from attention mask
        seq_lengths = attention_mask.sum(dim=1).to(torch.long)
        
        # Get the last valid token position for each sequence
        batch_indices = torch.arange(batch_size, device=device)
        last_positions = torch.clamp(seq_lengths - 1, min=0)  # Avoid negative indices
        
        # Extract activations at the last real token for each sequence
        # Shape: [batch_size, hidden_dim]
        last_token_activations = layer_activations[batch_indices, last_positions]

        # Apply the SAE encoder to get activations in the latent space
        # First reshape for encoder
        encoder_input = last_token_activations.to(sae.dtype)

        # Get SAE activations for the selected feature
        pre_activations = encoder_input @ sae.W_enc + sae.b_enc
        targets = pre_activations[:, feature_idx]

        # Make sure all zero activations for empty sequences (shouldn't happen, but just in case)
        targets = targets * (seq_lengths > 0).float()

        if good is not None:
            targets[~good] = -torch.finfo(targets.dtype).max

        return {
            "target": targets,
            "logits": logits,
            "one_hot": one_hot
        }
    return run