"""Shared helpers for the experiment scripts: paths, dataset streaming, small
curve summaries, JSON IO, and plotting.

Everything here is deliberately thin so the scripts and the Colab notebook can
share the same building blocks.
"""

import json
import os
import sys

import numpy as np

# Make `main`, `metrics` and `interventions` importable no matter where a script
# is launched from.
SRC_ROOT = os.path.abspath(os.path.dirname(__file__) + "/..")
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

REPO_ROOT = os.path.abspath(SRC_ROOT + "/../..")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
REFERENCE_DIR = os.path.join(RESULTS_DIR, "reference")


def ensure_dirs():
    for d in (RESULTS_DIR, FIGURES_DIR, REFERENCE_DIR):
        os.makedirs(d, exist_ok=True)


def get_hf_token(token=None):
    return (
        token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )


def load_imagenet(
    split="validation",
    streaming=True,
    token=None,
    shuffle=False,
    seed=0,
    buffer_size=2000,
    dataset_id="ILSVRC/imagenet-1k",
    fallback_id="benjamin-paine/imagenet-1k-256x256",
):
    """Stream ImageNet-1k from the Hub, yielding {'image': PIL, 'label': int}.

    The official split (`ILSVRC/imagenet-1k`) is gated. If your account has not
    accepted its terms, the load returns a 403 gated error. In that case this
    helper prints how to unlock the official split and falls back to an ungated
    repack (`benjamin-paine/imagenet-1k-256x256`) that carries the same image and
    label schema with the standard 0 to 999 label ordering, so every experiment,
    including fragility, still works. Pass fallback_id=None to disable the
    fallback and see the raw error.
    """
    from datasets import load_dataset

    token = get_hf_token(token)

    def _load(ds_id):
        ds = load_dataset(ds_id, split=split, streaming=streaming, token=token)
        if shuffle:
            ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
        return ds

    try:
        ds = _load(dataset_id)
        print(f"loaded {dataset_id} [{split}]")
        return ds
    except Exception as exc:  # gated / access / not found
        msg = str(exc).lower()
        gated = ("gated" in msg) or ("not found" in msg) or ("403" in msg) or ("access" in msg)
        if not (gated and fallback_id):
            raise
        print(
            f"Could not access '{dataset_id}' (it is gated for your account).\n"
            f"To use the official split, open\n"
            f"    https://huggingface.co/datasets/{dataset_id}\n"
            f"while logged in as the SAME account as your token, click "
            f"'Agree and access repository' (access is instant), then rerun.\n"
            f"Falling back to the ungated mirror '{fallback_id}' for now "
            f"(same schema and label ordering)."
        )
        ds = _load(fallback_id)
        print(f"loaded {fallback_id} [{split}]")
        return ds


def save_json(obj, path):
    ensure_dirs()
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path


def load_json(path):
    with open(path) as f:
        return json.load(f)


def summarize_curve(scores):
    """Compact summary of a per layer SSDC curve.

    peak        : max SSDC over depth.
    peak_layer  : depth at which the peak occurs.
    delta       : SSDC[1] - SSDC[0], the immediate recovery after the first block.
    decay       : peak - SSDC[last], how much SSDC falls from its peak by the end.
    final       : SSDC at the last layer.
    auc         : mean SSDC over depth.
    """
    s = np.asarray(scores, dtype=float)
    peak_layer = int(np.argmax(s))
    return {
        "peak": float(s.max()),
        "peak_layer": peak_layer,
        "delta": float(s[1] - s[0]) if s.size > 1 else 0.0,
        "decay": float(s.max() - s[-1]),
        "final": float(s[-1]),
        "auc": float(s.mean()),
    }


def plot_curves(curves, title, ylabel="SSDC", xlabel="Layer", save_path=None, ax=None, styles=None):
    """Plot several named per layer curves on one axis.

    curves : dict name -> list of per layer values (all the same length).
    styles : optional dict name -> matplotlib kwargs.
    """
    import matplotlib.pyplot as plt

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    styles = styles or {}
    for name, values in curves.items():
        xs = list(range(len(values)))
        ax.plot(xs, values, marker="o", markersize=3, label=name, **styles.get(name, {}))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axhline(0.0, color="0.7", linewidth=0.8, zorder=0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    if save_path and created:
        ensure_dirs()
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        print(f"saved {save_path}")
    return ax
