# ADAPT: Hybrid Prompt Optimization for LLM Feature Visualization

ADAPT finds short prompts that maximally activate a sparse autoencoder (SAE) latent. It starts from a diverse beam search and then alternates gradient-guided (GCG-style) token swaps with swaps sampled from the model's own next-token predictions. Diversity groups keep it exploring several different prompts, and a cross-entropy penalty that increases during the run pushes the prompts towards fluent text.

## Install

```bash
pip install -r requirements.txt
```

Works with any [sae_lens](https://github.com/jbloomAus/SAELens) `HookedSAETransformer` and SAE. The paper used Gemma 2 2B with Gemma Scope residual-stream SAEs (16k width, canonical). Gemma 2 is gated on Hugging Face, so accept its licence and run `huggingface-cli login` first.

## Usage

```python
from sae_lens import HookedSAETransformer, SAE
from ADAPT import ADAPT, OptimizationConfig

model = HookedSAETransformer.from_pretrained("gemma-2-2b", device="cuda")
sae = SAE.from_pretrained("gemma-scope-2b-pt-res-canonical", "layer_17/width_16k/canonical", device="cuda")

optimizer = ADAPT(model, sae)
result = optimizer.optimize(feat_id=12, config=OptimizationConfig())

optimizer.state.prompts, optimizer.state.activations   # final population and activations
```

`OptimizationConfig()` defaults to the paper's settings: 25 iterations, 10 prompts of 10 tokens, 3 diversity groups. Batch sizes adapt to the free GPU memory. Expect about 1.5 minutes per latent on an A40.

`quickstart.ipynb` runs one latent end to end. `latents.json` contains the 486 Gemma 2 latents evaluated in the paper, with their Neuronpedia top activations (for computing activation ratios) and the balanced and hardness-weighted subsets used for the ablations.

## Citation

```bibtex
@inproceedings{cardoso2026adapt,
  title     = {{ADAPT}: Hybrid Prompt Optimization for {LLM} Feature Visualization},
  author    = {Cardoso, Jo{\~a}o and Oliveira, Arlindo L. and Martins, Bruno},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
