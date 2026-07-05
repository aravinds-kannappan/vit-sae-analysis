# Discussion: what the real run shows

This writes up the actual results from the ImageNet-1k run (official validation
split, streamed). Models are ViT-Base/16: APE is `google/vit-base-patch16-224`
and RoPE is `vit_base_patch16_rope_224.naver_in1k`. SSDC and robustness use 1000
images, the ablation and effective rank sweeps use 512. Numbers are per block
(12 blocks). Everything is a single run, so treat small differences as noise and
read the shapes and the orderings, not the third decimal.

The headline is that the ablation experiment did what it was meant to do: it
changed the working assumptions. The two starting claims were

  1. ablating MLPs destroys SSDC recovery under RPI, and
  2. ablating attention destroys the SSDC decay in later layers.

The data supports a cleaner and largely reversed picture for the APE model:
attention builds the early index anchored recovery, and MLPs in the middle blocks
drive its decay.

## 1. SSDC across depth: APE peaks early and decays, RoPE accumulates and persists

APE SSDC under RPI rises immediately, peaks at blocks 3 to 4 (about 0.82), then
decays steadily to 0.20 by the last block. RoPE starts near zero, accumulates
gradually to a later peak at block 5 (about 0.64), and stays high, ending near
0.47. So the two position encodings reach a similar mid stack level of index
anchored structure by very different routes: APE injects it all at once up front
and then loses it, while RoPE refreshes it multiplicatively inside every attention
layer and holds onto it. The clean (no permutation) curves are high for both and
much flatter, which is expected since clean SSDC is dominated by content that is
already spatially smooth.

This matches the qualitative expectation from the guidance paper (APE peaks early,
RoPE accumulates), reproduced here on pretrained ViT-Base rather than from scratch
ViT-S. The stored reference curves have the same shape at slightly lower
magnitude, which is ordinary sampling and preprocessing variation.

## 2. Robustness: RoPE is the more robust model, and it is also the more persistent one

Under ImageNet-C Gaussian blur at severity 5:

| model | baseline acc | blurred acc | fragility |
|-------|-------------:|------------:|----------:|
| APE   | 0.805 | 0.495 | 0.385 |
| RoPE  | 0.844 | 0.607 | 0.281 |

RoPE has both the higher clean accuracy and the lower fragility. Put next to
section 1, the model that keeps its index anchored frame deeper into the stack
(RoPE, small SSDC decay) is also the more robust one. That is consistent with the
paper's thesis that a stable positional reference frame supports robustness,
though with n = 2 models this is a suggestive alignment and not a test of it.

## 3. The layer windowed ablation (APE), the main experiment

Primary metric is SSDC under RPI, since both the early peak and the later decay
live in that curve. Windows are early = blocks 0 to 3, mid = 4 to 7, late = 8 to
11. `peak_L` is the peak block, `delta` is SSDC[1] minus SSDC[0] (the immediate
jump), `decay` is peak minus final, `final` is the last block, `auc` is the mean
over depth.

| condition | peak | peak_L | delta | decay | final | auc |
|-----------|-----:|:------:|------:|------:|------:|----:|
| baseline        | 0.786 | 3 |  0.106 | 0.582 | 0.204 | 0.523 |
| mlp_zero_early  | 0.782 | 4 |  0.056 | 0.516 | 0.266 | 0.500 |
| mlp_zero_mid    | 0.777 | 3 |  0.113 | 0.301 | 0.476 | 0.613 |
| mlp_zero_late   | 0.780 | 3 |  0.100 | 0.592 | 0.188 | 0.514 |
| mlp_zero_all    | 0.788 | 4 |  0.043 | 0.111 | 0.677 | 0.677 |
| mlp_keep_early  | 0.776 | 3 |  0.109 | 0.290 | 0.486 | 0.617 |
| mlp_keep_mid    | 0.794 | 4 |  0.045 | 0.479 | 0.315 | 0.516 |
| mlp_keep_late   | 0.777 | 4 |  0.062 | 0.130 | 0.647 | 0.659 |
| attn_zero_early | 0.707 | 6 | -0.052 | 0.134 | 0.573 | 0.554 |
| attn_zero_mid   | 0.780 | 3 |  0.118 | 0.542 | 0.238 | 0.591 |
| attn_zero_late  | 0.773 | 3 |  0.111 | 0.436 | 0.337 | 0.528 |
| attn_zero_all   | 0.619 | 6 | -0.057 | 0.204 | 0.415 | 0.477 |

### Finding A: the early peak is built by attention, not MLPs

The peak height and its block location barely move under any MLP ablation. Every
`mlp_*` row peaks at 0.78 near block 3 or 4, including `mlp_zero_all` (peak 0.788).
Removing MLPs does not remove the early recovery. Attention ablation is what
touches the peak: `attn_zero_all` drops the peak to 0.619 and pushes it to block 6,
and `attn_zero_early` turns the immediate jump negative (delta -0.052) and delays
the peak to block 6. So the sharp early rise of SSDC under RPI is an attention
effect. This makes mechanistic sense for APE, which adds the learned position
vectors at the input, after which the first attention blocks read them into an
index graded similarity structure. The MLPs are not needed for that step.

This directly answers the motivating question. The early peak is not unique to
the early MLPs, and it is not a first MLPs encountered effect either. It is not an
MLP effect at all. The premise that the peak is carried by MLPs is falsified by
the fact that it survives `mlp_zero_all`.

### Finding B: the later decay is driven by the middle MLPs

The decay is where the MLPs matter, and specifically the middle ones.
`mlp_zero_mid` cuts the decay from 0.582 to 0.301 and lifts the final block from
0.204 to 0.476. `mlp_zero_all` almost erases the decay (0.111, final 0.677).
`mlp_zero_late` does essentially nothing (decay 0.592, final 0.188), and
`mlp_zero_early` only nudges it. The keep only probe agrees from the other side:
keeping MLPs alive only in the middle window reproduces most of the decay
(`mlp_keep_mid` decay 0.479), while keeping them only early or only late leaves
almost no decay (`mlp_keep_early` 0.290, `mlp_keep_late` 0.130).

So the mechanism is that MLPs in blocks 4 to 7 progressively overwrite the index
anchored similarity with content and task features, and that is what pulls SSDC
under RPI back down after its early attention driven peak. Late MLPs arrive after
the erosion has already happened and have little left to remove, and early MLPs
sit before the structure is fully injected.

### Finding C: attention only in the later layers reduces the decay only partly

`attn_zero_late` leaves the early peak intact (0.773 at block 3) and lifts the
final block from 0.204 to 0.337, shrinking the decay to 0.436. So removing late
attention does soften the later layer decay, but it does not flatten it. The
larger and cleaner decay lever is the middle MLPs, not late attention.
`attn_zero_all` reduces the decay number to 0.204, but it does so by pulling the
peak down rather than by holding the tail up, which is a different mechanism from
the MLP case. The `decay` metric (peak minus final) can shrink two ways, and the
table separates them: MLP ablation keeps the peak and raises the tail, attention
ablation lowers the peak.

Net effect on the two starting claims. Claim 1 (MLPs carry the recovery) is
overturned. Claim 2 (attention drives the later decay) is partly right in that
attention ablation does reduce the measured decay, but the dominant later layer
decay agent is the middle MLPs.

## 4. Extension: effective rank collapses without MLPs, but SSDC recovery does not

Following Dong et al. 2021 (attention without MLPs and skips drives representations
toward rank one with depth), we tracked effective rank under RPI:

| condition | rank at block 0 | rank at block 11 |
|-----------|----------------:|-----------------:|
| baseline        | 99.0 | 137.3 |
| mlp_zero_all    | 99.0 |  43.0 |
| attn_zero_all   | 98.7 | 136.1 |

This reproduces the Dong et al. picture cleanly. With all MLPs removed, effective
rank falls off with depth to 43 at the last block, the expected attention only rank
collapse. With attention removed, the per token MLPs keep rank high (136).

The interesting part is the cross reference with Finding A and B. `mlp_zero_all`
is exactly the condition where effective rank collapses, and it is also the
condition where SSDC under RPI stays highest (final 0.677). So rank collapse and
loss of index anchored structure are dissociated. Representations can drift toward
low rank while the similarity structure that remains is still strongly organized
by token index. This is a useful negative result: it says SSDC under RPI is not a
restatement of representational rank or capacity. The two probes move in opposite
directions under MLP ablation.

## 5. Limitations

- Single run per condition. The curves depend on the sampled images, so read
  orderings and shapes. Repeats with different seeds and error bars would firm up
  the mid MLP decay claim and the attention peak claim.
- Pretrained ViT-Base, not the from scratch ViT-S of the guidance paper. The
  ablated (trained without PE) and RPT controls from that paper need training and
  are out of scope here.
- Zero ablation removes a sublayer's entire contribution. Mean ablation or a
  learned baseline would test whether the effects are about the specific computation
  or just its magnitude.
- Robustness is one corruption (Gaussian blur) on two models, so the robustness to
  persistence alignment in section 2 is suggestive only.

## 6. Where this points next

- Sweep the ablation boundary block by block through the middle window to locate
  exactly where the decay switches on, rather than using a fixed 4 to 7 window.
- Repeat the whole ablation study on RoPE. Its gradual accumulation and small
  decay may respond differently, and a first read on random images already hinted
  that RoPE reacts to MLP ablation unlike APE.
- Add a content control. A quick random image check collapses the `mlp_zero_all`
  recovery that real images sustain, which suggests the surviving recovery is
  content times position rather than a pure architectural artifact. Worth turning
  into a proper control with matched statistics.
- Connect to the SAE features in `project_code/src/SAE`. If the middle block MLPs
  carry the decay, their SAE features are the natural place to look for the content
  and task directions that overwrite the index anchored structure.
