# Test notebook for EPO with SAE features
import torch
from sae_lens import HookedSAETransformer, SAE
from epo.epo import epo
from epo.runners import SAEFeatureRunner  # This will need to be implemented

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model
model = HookedSAETransformer.from_pretrained("google/gemma-2-2b").to('cuda')

# Load SAE for layer 16
sae, _, _ = SAE.from_pretrained(
    release="gemma-scope-2b-pt-res-canonical",
    sae_id=f"layer_16/width_16k/canonical",
    device="cuda",
)

# Create runner for feature 126
runner = SAEFeatureRunner(model, sae, feature_idx=126)

# Run EPO with default parameters 
history = epo(
    model=model,
    get_feature_and_logits=runner,
    seq_len=12  # Default sequence length
)

# Get final best prompt
best_tokens = history.ids[-1][history.keep[-1][0]]  # Get tokens for best member of final population
print("Final prompt:", model.tokenizer.decode(best_tokens))