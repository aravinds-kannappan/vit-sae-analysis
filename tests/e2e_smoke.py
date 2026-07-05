"""End to end smoke test against the REAL pretrained models.

Unlike tests/test_core.py (which needs nothing but torch), this downloads the two
ViT-Base checkpoints and runs the whole pipeline on a few random images: model
loading, the topology helpers, SSDC capture, RPI, component ablation, effective
rank, and the collate/predict path. It runs on CPU in float32, so the accuracy
numbers are meaningless (random images), but every code path that matters is
exercised. This is what catches library version breakage such as the transformers
ViT refactor or the BatchFeature half casting.

Needs: transformers, timm, and network access. Run:  python tests/e2e_smoke.py
"""

import os
import sys

import numpy as np
import torch
from PIL import Image

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "project_code", "src"))
sys.path.insert(0, SRC)

from main.load_models import load_ape, load_rope, num_prefix_tokens
from main.prep_data import prep_data
from main.model import predict
from metrics.ssdc import evaluate_ssdc
from metrics.effective_rank import evaluate_effective_rank
from interventions.ablation import AblationController, num_blocks


class FakeStream:
    """Mimics a HuggingFace streaming dataset: yields {'image': PIL, 'label': int}."""

    def __init__(self, n=6, size=256, seed=0):
        rng = np.random.default_rng(seed)
        self.items = [
            {
                "image": Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8)),
                "label": int(rng.integers(0, 1000)),
            }
            for _ in range(n)
        ]
        self.n_shards = 1

    def __iter__(self):
        return iter(self.items)

    def shard(self, num_shards, index):
        s = FakeStream(0)
        s.items = self.items[index::num_shards]
        return s

    def shuffle(self, **k):
        return self


def check(kind, loader, ds, N, B):
    print(f"\n===== {kind.upper()} =====")
    model, processor, source = loader()  # CPU, float32
    nb = num_blocks(model, source)
    print("num_blocks:", nb, "prefix_tokens:", num_prefix_tokens(model, source))

    kw = dict(number_images=N, batch_size=B, half=False, num_workers=0)

    clean, _ = evaluate_ssdc(model, processor, ds, source, RPI=False, **kw)
    rpi, _ = evaluate_ssdc(model, processor, ds, source, RPI=True, **kw)
    print("ssdc clean:", [round(x, 3) for x in clean])
    print("ssdc rpi  :", [round(x, 3) for x in rpi])
    assert len(clean) == nb and len(rpi) == nb

    with AblationController(model, source, "mlp", list(range(nb)), "zero"):
        abl, _ = evaluate_ssdc(model, processor, ds, source, RPI=True, **kw)
    print("rpi mlp0all:", [round(x, 3) for x in abl])
    assert len(abl) == nb

    with AblationController(model, source, "attn", [8, 9, 10, 11], "zero"):
        abl2, _ = evaluate_ssdc(model, processor, ds, source, RPI=True, **kw)
    assert len(abl2) == nb

    er = evaluate_effective_rank(model, processor, ds, source, RPI=False, **kw)
    print("effrank   :", [round(x, 1) for x in er])
    assert len(er) == nb

    dl_clean = prep_data(ds, processor, source, number_images=N, batch_size=B, half=False, num_workers=0)
    dl_blur = prep_data(ds, processor, source, corruption_type="Gaussian Blur", severity=5,
                        number_images=N, batch_size=B, half=False, num_workers=0)
    acc0, acc1 = predict(model, dl_clean, source, half=False), predict(model, dl_blur, source, half=False)
    print("acc clean/blur (meaningless on random imgs):", round(acc0, 3), round(acc1, 3))


def main():
    torch.manual_seed(0)
    ds = FakeStream(n=6)
    check("ape", load_ape, ds, 6, 3)
    check("rope", load_rope, ds, 6, 3)
    print("\nE2E SMOKE OK")


if __name__ == "__main__":
    main()
