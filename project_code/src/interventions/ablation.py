"""
Layer windowed component ablation for Vision Transformers.

This module provides AblationController, a context manager that zero ablates the
MLP or attention sublayers of a ViT at a chosen set of layer indices. It works
for both HuggingFace transformers ViT models (google/vit-base-patch16-224, the
APE model) and timm ViT models (vit_base_patch16_rope_224, the RoPE model) by
registering forward hooks on the relevant submodules.

A ViT block computes:

    x = x + attn(norm1(x))     # attention sublayer
    x = x + mlp(norm2(x))      # feedforward sublayer

Zero ablating a sublayer forces its output to zero, so the residual stream
passes through that sublayer untouched. This isolates the causal contribution of
one sublayer at one depth. Ablating a component at layer i changes the residual
stream from layer i onward, so its effect on a per layer metric shows up at
layer i+1 and beyond.

Two modes:
  - "zero"      : ablate the component at every layer listed in `layers`.
  - "keep_only" : ablate the component at every layer NOT in `layers`, so the
                  component survives only inside the given window. This is the
                  probe used to ask whether an effect is tied to specific depths
                  or simply to the first surviving sublayer the tokens meet.

The controller never touches weights. It only intercepts forward outputs, so it
is fully reversible and restores the model on exit.
"""

from contextlib import contextmanager
import os
import sys

import torch

# Topology helpers live in main.load_models so a single place knows how to find
# blocks and submodules across library versions (older transformers use
# vit.encoder.layer with a ViTOutput residual add, newer ones use vit.layers with
# a standalone block.mlp, timm uses model.blocks).
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
from main.load_models import get_vit_blocks, get_block_attention, get_block_mlp


def num_blocks(model, source):
    """Number of transformer blocks in the model."""
    return len(get_vit_blocks(model, source))


def resolve_layers(layers, n_layers, mode):
    """Turn a requested window plus a mode into the concrete set of layers that
    actually get their component zeroed."""
    requested = {int(l) for l in layers}
    universe = set(range(n_layers))
    unknown = requested - universe
    if unknown:
        raise ValueError(
            f"layer indices {sorted(unknown)} out of range for a {n_layers} layer model"
        )
    if mode == "zero":
        return requested
    if mode == "keep_only":
        return universe - requested
    raise ValueError(f"mode must be 'zero' or 'keep_only', got {mode!r}")


def _attn_ablation_hook(module, inputs, output):
    """Zero the attention sublayer output.

    transformers ViTAttention returns a tuple whose first element is the context
    tensor; timm attention returns a bare tensor. Handle both.
    """
    if isinstance(output, tuple):
        zeroed = torch.zeros_like(output[0])
        return (zeroed,) + tuple(output[1:])
    return torch.zeros_like(output)


def _zero_output_hook(module, inputs, output):
    """Zero a sublayer output whose result is added to the residual outside the
    module (timm MLP, and the standalone MLP in newer transformers)."""
    return torch.zeros_like(output)


def _residual_mlp_hook(module, inputs, output):
    """Remove an older transformers MLP whose module adds the residual internally.

    ViTOutput.forward(hidden_states, input_tensor) returns dense(hidden_states) +
    input_tensor. Returning input_tensor alone keeps the residual and drops the
    feedforward update.
    """
    if len(inputs) >= 2 and torch.is_tensor(inputs[1]):
        return inputs[1]
    raise RuntimeError(
        "Unexpected MLP output signature; cannot ablate MLP in 'residual' mode. "
        "Expected ViTOutput.forward(hidden_states, input_tensor)."
    )


class AblationController:
    """Context manager that zero ablates a component at chosen layers.

    Parameters
    ----------
    model : the loaded ViT (transformers ViTForImageClassification or a timm ViT).
    source : "transformers" or "timm".
    component : "attn" or "mlp".
    layers : iterable of layer indices. Meaning depends on `mode`.
    mode : "zero" (ablate the listed layers) or "keep_only" (ablate everything
        except the listed layers).

    Usage
    -----
    >>> with AblationController(model, "transformers", "mlp", [0, 1, 2, 3]):
    ...     scores, _ = evaluate_ssdc(model, processor, dataset, "transformers", RPI=True)
    """

    def __init__(self, model, source, component, layers, mode="zero"):
        if component not in ("attn", "mlp"):
            raise ValueError(f"component must be 'attn' or 'mlp', got {component!r}")
        self.model = model
        self.source = source
        self.component = component
        self.mode = mode
        self.n_layers = num_blocks(model, source)
        self.ablated_layers = resolve_layers(layers, self.n_layers, mode)
        self.handles = []

    def __enter__(self):
        blocks = get_vit_blocks(self.model, self.source)
        for i, blk in enumerate(blocks):
            if i not in self.ablated_layers:
                continue
            if self.component == "attn":
                target = get_block_attention(blk, self.source)
                handle = target.register_forward_hook(_attn_ablation_hook)
            else:  # mlp
                mlp_mod, mode = get_block_mlp(blk, self.source)
                hook = _zero_output_hook if mode == "zero" else _residual_mlp_hook
                handle = mlp_mod.register_forward_hook(hook)
            self.handles.append(handle)
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        return False

    def summary(self):
        return {
            "component": self.component,
            "mode": self.mode,
            "requested_or_kept": self.mode,
            "ablated_layers": sorted(self.ablated_layers),
            "n_layers": self.n_layers,
        }


@contextmanager
def no_ablation():
    """A do nothing context manager, useful as the baseline condition so that
    experiment loops can treat baseline and ablated runs uniformly."""
    yield None
