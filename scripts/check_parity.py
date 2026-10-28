"""Action parity: every variant vs stock PyTorch eager, on the same observations and noise.

JAX (the original openpi implementation) is included as a cross-framework reference
for how large bf16 implementation differences are.

Observations: with --obs-dir, N requests recorded from real LIBERO rollouts
(serve.py --record), evenly spaced; otherwise the 40 LIBERO task instructions with
random images and states. Arm dimensions (0-5) and the gripper (6, executed by its
sign) are reported separately. Raw actions are saved next to <out.json>.

Usage: python scripts/check_parity.py <out.json> [--obs-dir DIR] [--n-obs N] [--noises K] [--variants V ...]
                                      [--modes M ...] [--no-jax] [--num-steps N]
"""

import argparse
import gc
import json
import os
import pathlib
import re

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # JAX and PyTorch share one GPU

import numpy as np
import torch
from openpi.policies import libero_policy
from openpi.policies import policy_config
from openpi.training import config as _config
import pi05_accel

# Task instructions for the random-observation mode; run from the openpi checkout or set OPENPI_DIR.
LIBERO_BDDL = pathlib.Path(os.environ.get("OPENPI_DIR", ".")) / "third_party/libero/libero/libero/bddl_files"


def observations(n, seed=0):
    names = sorted({p.stem for p in LIBERO_BDDL.glob("libero_*/*.bddl") if p.parent.name != "libero_90"})
    prompts = [re.sub(r"^[A-Z_]+_SCENE\d+_", "", s).replace("_", " ") for s in names]
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        obs = libero_policy.make_libero_example()
        obs["observation/image"] = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
        obs["observation/wrist_image"] = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
        obs["observation/state"] = rng.uniform(-1, 1, 8)
        obs["prompt"] = prompts[i % len(prompts)]
        out.append((obs, rng.standard_normal((10, 32)).astype(np.float32)))
    return out


def recorded_observations(obs_dir, n, noises=1, seed=0):
    rng = np.random.default_rng(seed)
    return [
        (obs, rng.standard_normal((10, 32)).astype(np.float32))
        for obs in pi05_accel.load_recorded(obs_dir, n)
        for _ in range(noises)
    ]


def compare(a, ref, noises):
    """Arm (6 dims) and gripper differences, over all 10 actions and the first 5 (what the LIBERO client executes)."""
    d = np.abs(a - ref)
    per_obs = d[..., :6].reshape(-1, noises, *d.shape[1:2], 6).max(axis=(1, 2, 3))  # worst over noises, actions, dims
    return {
        "arm_max_abs": float(d[..., :6].max()),
        "arm_mean_abs": float(d[..., :6].mean()),
        "first5_arm_max_abs": float(d[:, :5, :6].max()),
        "first5_arm_mean_abs": float(d[:, :5, :6].mean()),
        "grip_max_abs": float(d[..., 6].max()),
        "grip_sign_agree": float((np.sign(a[..., 6]) == np.sign(ref[..., 6])).mean()),
        "first5_grip_sign_agree": float((np.sign(a[:, :5, 6]) == np.sign(ref[:, :5, 6])).mean()),
        "worst_obs": [int(i) for i in np.argsort(per_obs)[::-1][:5]],
        "worst_obs_arm_max": [float(per_obs[i]) for i in np.argsort(per_obs)[::-1][:5]],
        "per_obs_arm_max_median": float(np.median(per_obs)),
        "bit_identical": bool((a == ref).all()),
    }


def run(policy, data):
    actions = np.stack([policy.infer(obs, noise=noise)["actions"] for obs, noise in data])
    del policy  # free weights and CUDA graph pools before the next case
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  (allocated after cleanup: {torch.cuda.memory_allocated() / 2**30:.2f} GiB)")
    return actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--obs-dir", default=None)
    ap.add_argument("--n-obs", type=int, default=40)
    ap.add_argument("--noises", type=int, default=1, help="noise draws per observation")
    ap.add_argument("--variants", nargs="+", default=list(pi05_accel.VARIANTS))
    ap.add_argument("--modes", nargs="+", default=["eager", "max-autotune"], help="eager and/or torch.compile modes")
    ap.add_argument("--no-jax", action="store_true")
    ap.add_argument("--num-steps", type=int, default=10, help="denoising steps for non-stock variants (behaviour-changing)")
    args = ap.parse_args()
    data = recorded_observations(args.obs_dir, args.n_obs, args.noises) if args.obs_dir else observations(args.n_obs)
    assert args.obs_dir or args.noises == 1, "--noises needs --obs-dir"

    ref = run(pi05_accel.make_policy("stock", None), data)
    results = {}
    cases = [(v, None if m == "eager" else m) for v in args.variants for m in args.modes]
    for variant, mode in cases:
        steps = {} if variant == "stock" else {"num_steps": args.num_steps}
        results[f"{variant}/{mode or 'eager'}"] = run(pi05_accel.make_policy(variant, mode, **steps), data)
    if not args.no_jax:
        jax_dir = pi05_accel.CKPT.parent / "pi05_libero"
        results["jax"] = run(policy_config.create_trained_policy(_config.get_config("pi05_libero"), jax_dir), data)

    report = {
        "provenance": pi05_accel.provenance(__file__),
        "n_obs": args.n_obs,
        "noises_per_obs": args.noises,
        "obs": args.obs_dir or "random",
        "ref": "stock/eager",
        "ref_mean_abs_arm": float(np.abs(ref[..., :6]).mean()),
        "cases": {name: compare(a, ref, args.noises) for name, a in results.items()},
    }
    if "stock/max-autotune" in results:  # the deployment-relevant reference
        ref2 = results["stock/max-autotune"]
        report["vs_stock_compiled"] = {name: compare(a, ref2, args.noises) for name, a in results.items()}
    for title, block in [("vs stock eager", report["cases"]), ("vs stock compiled", report.get("vs_stock_compiled", {}))]:
        print(f"== {title}: arm max/mean (all 10 | first 5 executed), gripper sign agreement, worst obs")
        for name, c in block.items():
            print(f"{name:42s} {c['arm_max_abs']:.4f}/{c['arm_mean_abs']:.5f} | {c['first5_arm_max_abs']:.4f}/"
                  f"{c['first5_arm_mean_abs']:.5f}  grip {c['grip_sign_agree']:.4f}  worst {c['worst_obs'][:3]}"
                  f"{'  bit-identical' if c['bit_identical'] else ''}")
    print(f"reference mean |arm action| {report['ref_mean_abs_arm']:.4f}")
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    np.savez_compressed(out.with_suffix(".npz"), ref=ref, **{k.replace("/", "_"): v for k, v in results.items()})


if __name__ == "__main__":
    main()
