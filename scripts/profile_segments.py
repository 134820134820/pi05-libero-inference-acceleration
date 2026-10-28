"""Per-segment latency of one pi05_libero inference (PyTorch).

Wraps the stock openpi methods instead of re-implementing them, so the timed code
path is exactly the served one. Each segment records CUDA events (asynchronous,
no extra syncs) for GPU time and perf_counter for CPU wall time.

With --compile MODE the model internals are not wrapped (that would break the
compiled graph); only host-side segments and the whole model call are timed, and
dynamo graph breaks are counted. --trace adds a torch.profiler pass that counts
CUDA kernels and their summed busy time per inference and saves a Chrome trace.

Usage: python scripts/profile_segments.py <out.json> [--variant V] [--compile MODE] [--trace] [--iters N] [--warmup N]
                                         [--obs-dir DIR --n-obs N] [--num-steps N] [--lang-bucket N]
  V is "stock" (default), "+"-joined pi05_accel.FEATURES, or jax: openpi's original
  JAX implementation (XLA-jitted), timed end to end only.
"""

import argparse
import collections
import json
import pathlib
import time

import numpy as np
import torch
from openpi.policies import libero_policy
import pi05_accel


class Timer:
    """Collects (segment -> [cpu_ms, gpu events]) for one inference."""

    def __init__(self):
        self.cpu = collections.defaultdict(float)
        self.events = collections.defaultdict(list)

    def wrap(self, obj, name, label, gpu=True):
        fn = getattr(obj, name)

        def timed(*args, **kwargs):
            seg = label(args, kwargs) if callable(label) else label
            if gpu:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            self.cpu[seg] += (time.perf_counter() - t0) * 1e3
            if gpu:
                end.record()
                self.events[seg].append((start, end))
            return out

        setattr(obj, name, timed)

    def collect(self):
        torch.cuda.synchronize()
        gpu = {k: sum(s.elapsed_time(e) for s, e in v) for k, v in self.events.items()}
        row = {f"cpu_ms/{k}": v for k, v in self.cpu.items()} | {f"gpu_ms/{k}": v for k, v in gpu.items()}
        self.cpu.clear()
        self.events.clear()
        return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--variant", default="stock", help='"stock", "jax", or "+"-joined pi05_accel.FEATURES')
    ap.add_argument("--lang-bucket", type=int, default=None, help="prune: language bucket (default: pi05_accel's)")
    ap.add_argument("--num-steps", type=int, default=10, help="denoising steps (static variants; behaviour-changing)")
    ap.add_argument("--compile", default=None, help="torch.compile mode, e.g. max-autotune; default eager")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--obs-dir", default=None, help="time over recorded observations (serve.py --record) instead")
    ap.add_argument("--n-obs", type=int, default=200)
    args = ap.parse_args()
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    t = Timer()
    if args.variant == "jax":
        from openpi.policies import policy_config
        from openpi.training import config as _config

        policy = policy_config.create_trained_policy(
            _config.get_config("pi05_libero"), pi05_accel.CKPT.parent / "pi05_libero"
        )
    else:
        extra = {"num_steps": args.num_steps} | ({} if args.lang_bucket is None else {"lang_bucket": args.lang_bucket})
        policy = pi05_accel.make_policy(args.variant, args.compile, **extra)
        model = policy._model  # noqa: SLF001
        pwe = model.paligemma_with_expert
        # Whole model call; Policy.infer copies the result to the host right after it.
        t.wrap(policy, "_sample_actions", "model/total")
    # Host side (CPU only): openpi input/output transforms.
    t.wrap(policy, "_input_transform", "host/input_transform", gpu=False)
    t.wrap(policy, "_output_transform", "host/output_transform", gpu=False)
    if args.variant != "jax" and args.compile is None:  # model internals, eager only
        t.wrap(model, "_preprocess_observation", "model/preprocess")
        t.wrap(pwe, "embed_image", "model/vision_siglip")
        t.wrap(pwe, "embed_language_tokens", "model/lang_embed")
        t.wrap(pwe, "forward", lambda a, kw: "model/prefix_gemma" if kw["inputs_embeds"][0] is not None else "model/expert_fwd")
        t.wrap(model, "denoise_step", "model/denoise_steps")

    if args.obs_dir:  # real LIBERO observations, cycled
        obs_list = pi05_accel.load_recorded(args.obs_dir, args.n_obs)
    else:  # one fixed synthetic observation (shapes are what matters for the compute)
        np.random.seed(0)  # make_libero_example draws its images from the global RNG
        obs = libero_policy.make_libero_example()
        obs["observation/state"] = np.random.default_rng(0).random(8)
        obs["prompt"] = "put the bowl on the plate"
        obs_list = [obs]
    obs = obs_list[0]

    rows = []
    torch.cuda.reset_peak_memory_stats()
    t_first = time.perf_counter()
    policy.infer(obs)  # first call: compilation (if any) and allocator warm-up
    first_call_s = time.perf_counter() - t_first
    t.collect()
    for i in range(args.warmup + args.iters):
        t0 = time.perf_counter()
        policy.infer(obs_list[i % len(obs_list)])
        e2e = (time.perf_counter() - t0) * 1e3
        row = t.collect() | {"cpu_ms/infer_e2e": e2e}
        if i >= args.warmup:
            rows.append(row)

    keys = sorted(rows[0])
    summary = {k: {"p50": float(np.median([r[k] for r in rows])), "p95": float(np.percentile([r[k] for r in rows], 95))} for k in keys}
    result = {
        "provenance": pi05_accel.provenance(__file__),
        "obs": args.obs_dir or "synthetic",
        "variant": args.variant,
        "lang_bucket": args.lang_bucket,
        "num_steps": args.num_steps,
        "compile": args.compile,
        "first_call_s": first_call_s,
        "iters": args.iters,
        "warmup": args.warmup,
        # torch's allocator only; meaningless for JAX, which manages its own memory.
        "peak_mem_gib": None if args.variant == "jax" else torch.cuda.max_memory_allocated() / 2**30,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "summary": summary,
        "rows": rows,
    }
    if args.compile is not None:
        from torch._dynamo.utils import counters

        result["graph_breaks"] = dict(counters["graph_break"])
    if args.trace:
        result["trace"] = kernel_stats(policy, obs, pathlib.Path(args.out).with_suffix(".trace.json.gz"))
    pathlib.Path(args.out).write_text(json.dumps(result, indent=1))
    for k in keys:
        print(f"{k:32s} p50 {summary[k]['p50']:8.2f}  p95 {summary[k]['p95']:8.2f}")
    print(f"peak mem {result['peak_mem_gib']} GiB; first call {first_call_s:.1f} s")
    for k in ("graph_breaks", "trace"):
        if k in result:
            print(k, result[k])


def kernel_stats(policy, obs, trace_path, n=5):
    """CUDA kernel count and busy time (union of kernel intervals) per inference; copies and memsets excluded."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t0 = time.perf_counter()
        for _ in range(n):
            policy.infer(obs)
        wall_ms = (time.perf_counter() - t0) * 1e3 / n
    kernels = sorted(
        (e.time_range.start, e.time_range.end)
        for e in prof.events()
        if e.device_type == torch.autograd.DeviceType.CUDA and not e.name.startswith(("Memcpy", "Memset"))
    )
    busy, cur_s, cur_e = 0.0, None, None
    for s, e in kernels:  # merge overlapping intervals (us)
        if cur_e is None or s > cur_e:
            busy += (cur_e - cur_s) if cur_e is not None else 0.0
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += (cur_e - cur_s) if cur_e is not None else 0.0
    prof.export_chrome_trace(str(trace_path))
    return {"kernels_per_infer": len(kernels) / n, "gpu_busy_ms": busy / 1e3 / n, "wall_ms": wall_ms}


if __name__ == "__main__":
    main()
