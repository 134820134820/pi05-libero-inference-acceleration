"""Faster pi05 inference on top of stock openpi (PyTorch), without editing openpi.

Independent changes, selected as "+"-joined features (e.g. "prune+static+cond");
all are exact in real arithmetic:

prune   Drop prefix tokens that every query masks out anyway: padding images
        (image_mask False, e.g. LIBERO's missing right-wrist camera) and language
        padding beyond a bucketed length. Masked keys get -2.38e38 before softmax
        (weight exactly 0) and position ids are cumsum(pad_mask) - 1, so the kept tokens
        see the same keys at the same positions. Decided on the host, before the
        tensors reach the GPU, so shapes stay static per bucket.

static  Denoise with a fixed number of steps whose float32 time values are
        precomputed exactly as the stock `while time >= -dt/2` loop produces them,
        and skip the eval-mode preprocessing (a no-op for 224x224 inputs whose
        set.issubset check breaks the graph). sample_actions then traces as a single
        graph, so torch.compile can replay it as one CUDA graph instead of returning
        to Python, and syncing on `time`, every step.

cond    (needs static) In pi05 the expert's adaptive RMSNorms are conditioned only
        on the timestep: every norm computes dense(time_mlp(t)), an FP32 1024->3072
        layer, i.e. ~12.6 MB of weights read per norm per step (37 norms x 10 steps).
        With a fixed schedule those are constants, so each norm's dense layer is
        replaced by a table of its outputs at the scheduled timesteps, computed once
        with the stock code; the norms receive the step index instead.

batchimg  Encode all camera images with SigLIP as one batch instead of one call
        per image (embed_prefix otherwise unchanged).

aten    Compile setting, not a model change: max-autotune considers only ATen (cuBLAS)
        kernels for matrix multiplies instead of also Triton templates. Needed by the
        fp8 features (Inductor in torch 2.7.1 fails to lower the activation-scale
        reduction, aten.amin, when it reads a Triton GEMM template output); measured on
        its own as the control for them.

sdpa    Run the Gemma attention (prefix and expert) through PyTorch SDPA instead of
        the eager matmul-softmax-matmul that openpi forces. Each attention module
        gets its own config copy set to "sdpa", so openpi's per-call assignment of
        "eager" to the model-level config no longer reaches it. openpi always passes
        a full 4D additive mask, so the attention stays non-causal with the same mask;
        SDPA needs it in the query dtype, so it is built in bf16 (-2.38e38 is still
        finite there, so fully masked padding rows stay finite as in eager).

Tier 2, engineering trade-offs that change the outputs:

fp8 / fp8vit / fp8expert  torchao FP8 (e4m3) dynamic activation + weight quantization,
        per-tensor scales (per-row scaling is not supported on sm_120 in torch 2.7.1),
        of the bf16 Linear layers in the Gemma prefix language model / the SigLIP
        vision tower / the action expert respectively.
int8 / int8vit / int8expert  torchao INT8 dynamic quantization (W8A8: per-token
        activation scales, per-channel weight scales) of the same layer groups. Works
        with the default max-autotune; FP8 needs `aten`, whose ATen-only GEMMs cost
        3.75 ms on the unquantized layers (measured). torchao is told not to change the global
        Inductor config (set_inductor_config=False): by default it does, which also
        changes how the unquantized layers compile and leaks into later policies.
        The quantized weights stay torchao tensor subclasses, which adds host time per
        call; torchao.utils.unwrap_tensor_subclass fails to trace under
        dynamo with torchao 0.11 + torch 2.7.1, so that overhead stays.
num_steps != 10 (make_policy argument): fewer Euler denoising steps.

The stock building blocks (embed_prefix, the PaliGemma/expert forward, denoise_step)
are called unchanged.
"""

import copy
import dataclasses
import hashlib
import logging
import math
import os
import pathlib
import sys
import time

import numpy as np
import torch
from openpi.models_pytorch import pi0_pytorch
from openpi.models_pytorch import preprocessing_pytorch
from openpi.policies import policy_config
from openpi.training import config as _config

CKPT = pathlib.Path(os.environ.get("OPENPI_DATA_HOME", "~/.cache/openpi")).expanduser() / (
    "openpi-assets/checkpoints/pi05_libero_pytorch"
)
FEATURES = ("prune", "static", "cond", "batchimg", "aten", "sdpa", "fp8", "fp8vit", "fp8expert",
            "int8", "int8vit", "int8expert")
# The measured combinations; "prune+static+cond+batchimg" is the final tier-1 configuration.
VARIANTS = ("stock", "static", "prune", "prune+static", "prune+static+cond", "prune+static+cond+batchimg")


def parse_variant(variant):
    """"stock" -> set(); otherwise the set of "+"-joined FEATURES."""
    feats = set() if variant == "stock" else set(variant.split("+"))
    assert feats <= set(FEATURES), f"unknown feature in {variant!r}"
    assert "cond" not in feats or "static" in feats, "cond needs static (a fixed schedule)"
    return feats


def provenance(script):
    """What produced a result: command line, content hashes of this module and the calling script, versions."""
    sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:12]  # noqa: E731
    return {
        "argv": sys.argv,
        "pi05_accel_sha256": sha(__file__),
        "script_sha256": sha(script),
        "torch": torch.__version__,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def load_recorded(obs_dir, n):
    """n observations evenly spaced over a serve.py --record directory (openpi PolicyRecorder files)."""
    steps = sorted(pathlib.Path(obs_dir).glob("step_*.npy"), key=lambda p: int(p.stem.split("_")[1]))
    out = []
    for i in np.linspace(0, len(steps) - 1, min(n, len(steps))).round().astype(int):
        rec = np.load(steps[i], allow_pickle=True).item()
        out.append({k.removeprefix("inputs/"): v for k, v in rec.items() if k.startswith("inputs/")})
    return out


def denoise_schedule(num_steps):
    """The float32 time values the stock loop visits, and its float32 dt."""
    dt = np.float32(-1.0 / num_steps)
    t, times = np.float32(1.0), []
    while t >= -dt / 2:
        times.append(float(t))
        t = np.float32(t + dt)
    return times, float(dt)


def prune_inputs(inputs, lang_bucket):
    """Host side, after openpi's input transforms: drop padding images and language padding.

    One observation at a time (openpi's Policy.infer adds the batch dimension later).
    """
    keep = [k for k in inputs["image"] if inputs["image_mask"][k]]
    inputs["image"] = {k: inputs["image"][k] for k in keep}
    inputs["image_mask"] = {k: inputs["image_mask"][k] for k in keep}
    mask = inputs["tokenized_prompt_mask"]
    n = int(mask.sum())
    assert mask[:n].all(), "expected right-padded prompt"
    length = min(len(mask), math.ceil(n / lang_bucket) * lang_bucket)
    inputs["tokenized_prompt"] = inputs["tokenized_prompt"][:length]
    inputs["tokenized_prompt_mask"] = mask[:length]
    return inputs


class _StepTable(torch.nn.Module):
    """Stands in for an adaptive norm's dense layer: row i is dense(cond) at denoise step i."""

    def __init__(self, table):
        super().__init__()
        self.register_buffer("table", table)

    def forward(self, step):  # step: (batch, 1) float step index, passed where cond was
        return self.table[step[:, 0].long()]


@torch.no_grad()
def precompute_conditioning(model, times):
    """Replace the expert's adaptive-norm dense layers by step tables (the `cond` feature)."""
    device = model.action_in_proj.weight.device
    dummy_actions = torch.zeros(1, model.config.action_horizon, model.config.action_dim, device=device)
    # pi05's embed_suffix ignores state; its 4th output is the timestep-only adaRMS condition.
    conds = [model.embed_suffix(None, dummy_actions, torch.full((1,), t, device=device))[3] for t in times]
    norms = [m for m in model.paligemma_with_expert.gemma_expert.modules() if getattr(m, "dense", None) is not None]
    for norm in norms:
        norm.dense = _StepTable(torch.cat([norm.dense(c) for c in conds]))
        norm.cond_dim = 1
    times_t = torch.tensor(times, dtype=torch.float32, device=device)
    embed_suffix = model.embed_suffix

    def embed_suffix_with_step(state, noisy_actions, timestep):
        # The stock time MLP still runs here, but its output is unused (dead code once compiled).
        # The nearest scheduled time is looked up; this is only equivalent for timesteps on the
        # schedule, which make_policy guarantees (cond requires static, same schedule).
        embs, pad_masks, att_masks, _ = embed_suffix(state, noisy_actions, timestep)
        step = (times_t[None, :] - timestep[:, None]).abs().argmin(dim=1, keepdim=True).to(torch.float32)
        return embs, pad_masks, att_masks, step

    model.embed_suffix = embed_suffix_with_step
    return len(norms)


def embed_prefix_batched(model, images, img_masks, lang_tokens, lang_masks):
    """PI0Pytorch.embed_prefix, with all camera images through SigLIP in one call (the `batchimg` feature)."""
    bsize = images[0].shape[0]
    img_embs = model.paligemma_with_expert.embed_image(torch.cat(images, dim=0)).split(bsize)
    embs, pad_masks, att_masks = [], [], []
    for img_emb, img_mask in zip(img_embs, img_masks, strict=True):
        embs.append(img_emb)
        pad_masks.append(img_mask[:, None].expand(bsize, img_emb.shape[1]))
        att_masks += [0] * img_emb.shape[1]
    lang_emb = model.paligemma_with_expert.embed_language_tokens(lang_tokens)
    embs.append(lang_emb * math.sqrt(lang_emb.shape[-1]))
    pad_masks.append(lang_masks)
    att_masks += [0] * lang_emb.shape[1]
    embs = torch.cat(embs, dim=1)
    pad_masks = torch.cat(pad_masks, dim=1)
    att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)[None, :].expand(bsize, -1)
    return embs, pad_masks, att_masks


def use_sdpa(model):
    """Give every Gemma attention module its own config with SDPA (the `sdpa` feature)."""
    prepare = model._prepare_attention_masks_4d  # noqa: SLF001
    model._prepare_attention_masks_4d = lambda m: prepare(m).to(torch.bfloat16)  # noqa: SLF001
    n = 0
    for m in model.paligemma_with_expert.modules():
        if type(m).__name__ == "GemmaAttention":
            m.config = copy.copy(m.config)
            m.config._attn_implementation = "sdpa"  # noqa: SLF001
            n += 1
    return n


def quantize_linears(module, kind):
    """torchao dynamic quantization of the bf16 Linear layers under `module`: kind "fp8" or "int8"."""
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig
    from torchao.quantization import PerTensor
    from torchao.quantization import quantize_

    config = {
        "fp8": lambda: Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor(), set_inductor_config=False),
        "int8": lambda: Int8DynamicActivationInt8WeightConfig(set_inductor_config=False),
    }[kind]()

    def pick(m, _fqn):
        return isinstance(m, torch.nn.Linear) and m.weight.dtype == torch.bfloat16 and m.in_features % 16 == 0 \
            and m.out_features % 16 == 0

    n = sum(pick(m, "") for m in module.modules())
    quantize_(module, config, filter_fn=pick)
    logging.info("quantized %d Linear layers (%s) under %s", n, kind, type(module).__name__)
    return n


def _stock_preprocess(observation):
    """PI0Pytorch._preprocess_observation over whichever images are present.

    Unpacking to a tuple inside this function matters under torch.compile: the stock
    preprocessing returns an instance of a class defined inside it (a new type on
    every call) and graph-breaks; resuming in the caller with that object fails the
    type guard each call, hits the recompile limit, and runs eagerly. The stock
    model unpacks the same way, so its graph breaks are reproduced, nothing more.
    """
    obs = preprocessing_pytorch.preprocess_observation_pytorch(
        observation, train=False, image_keys=tuple(observation.images)
    )
    return (
        list(obs.images.values()),
        list(obs.image_masks.values()),
        obs.tokenized_prompt,
        obs.tokenized_prompt_mask,
        obs.state,
    )


@torch.no_grad()
def sample_actions(model: pi0_pytorch.PI0Pytorch, device, observation, noise=None, num_steps=10, *, schedule=None):
    """pi0_pytorch.PI0Pytorch.sample_actions; schedule=denoise_schedule(num_steps) selects `static`.

    The schedule is passed in rather than computed here: computed inside a compiled
    function, its numpy float32 loop is itself traced and breaks the graph.
    """
    static = schedule is not None
    bsize = observation.state.shape[0]
    if noise is None:
        noise = model.sample_noise((bsize, model.config.action_horizon, model.config.action_dim), device)

    if static:
        images, img_masks = list(observation.images.values()), list(observation.image_masks.values())
        lang_tokens, lang_masks, state = observation.tokenized_prompt, observation.tokenized_prompt_mask, observation.state
    else:
        images, img_masks, lang_tokens, lang_masks, state = _stock_preprocess(observation)

    # Prefix KV cache: identical to stock.
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    prefix_att_2d_masks = pi0_pytorch.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)  # noqa: SLF001
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )

    x_t = noise
    if static:
        times, dt = schedule
        for t in times:
            timestep = torch.full((bsize,), t, dtype=torch.float32, device=device)
            v_t = model.denoise_step(state, prefix_pad_masks, past_key_values, x_t, timestep)
            x_t = x_t + dt * v_t
    else:  # stock loop
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            v_t = model.denoise_step(state, prefix_pad_masks, past_key_values, x_t, time.expand(bsize))
            x_t = x_t + dt * v_t
            time += dt
    return x_t


def make_policy(variant="stock", compile_mode="max-autotune", checkpoint=CKPT, lang_bucket=64, num_steps=10, **kwargs):
    """A pi05_libero policy. variant: "stock" or "+"-joined FEATURES; compile_mode None means eager.

    num_steps other than openpi's 10 changes the Euler discretisation, i.e. the model's
    behaviour (an engineering trade-off, not an exact optimisation); for variants it
    needs `static`, for stock it is passed to openpi's sample_actions (used here only to
    measure the per-step cost the same way for every configuration).
    """
    feats = parse_variant(variant)
    assert num_steps == 10 or not feats or "static" in feats, "num_steps needs the static schedule"
    cfg = _config.get_config("pi05_libero")
    stock_compile = compile_mode if not feats else None
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=stock_compile))
    if not feats and num_steps != 10:  # stock sample_actions takes num_steps directly
        kwargs["sample_kwargs"] = {"num_steps": num_steps}
    policy = policy_config.create_trained_policy(cfg, checkpoint, **kwargs)
    if not feats:
        return policy

    model = policy._model  # noqa: SLF001
    if "prune" in feats:
        transform = policy._input_transform  # noqa: SLF001
        policy._input_transform = lambda x: prune_inputs(transform(x), lang_bucket)  # noqa: SLF001
    schedule = denoise_schedule(num_steps) if "static" in feats else None
    if "cond" in feats:
        precompute_conditioning(model, schedule[0])
    if "batchimg" in feats:
        model.embed_prefix = lambda *a: embed_prefix_batched(model, *a)
    if "sdpa" in feats:
        use_sdpa(model)
    pwe = model.paligemma_with_expert
    groups = {"": pwe.paligemma.language_model, "vit": pwe.paligemma.vision_tower, "expert": pwe.gemma_expert}
    for kind in ("fp8", "int8"):
        for suffix, module in groups.items():
            if kind + suffix in feats:
                quantize_linears(module, kind)

    def fn(device, observation, noise=None):
        return sample_actions(model, device, observation, noise, schedule=schedule)

    if not compile_mode:
        policy._sample_actions = fn  # noqa: SLF001
    elif "aten" in feats or feats & {"fp8", "fp8vit", "fp8expert"}:
        # Per-compile options, so the GEMM backend choice does not leak into other policies in the process.
        from torch._inductor import list_mode_options

        options = list_mode_options(compile_mode) | {"max_autotune_gemm_backends": "ATEN"}
        policy._sample_actions = torch.compile(fn, options=options)  # noqa: SLF001
    else:
        policy._sample_actions = torch.compile(fn, mode=compile_mode)  # noqa: SLF001
    return policy


if __name__ == "__main__":
    # Self-check: the precomputed schedule matches the stock float32 loop.
    times, dt = denoise_schedule(10)
    assert len(times) == 10 and times[0] == 1.0 and dt == float(np.float32(-0.1)), (times, dt)
    t = torch.tensor(1.0)
    for expected in times:
        assert t.item() == expected
        t += torch.tensor(dt)
    assert not (t >= -torch.tensor(dt) / 2)
    print("schedule ok:", [f"{x:.8f}" for x in times])
