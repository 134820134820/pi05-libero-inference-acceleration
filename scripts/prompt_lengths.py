"""Valid language-token counts of recorded LIBERO requests, through openpi's own input transform.

Usage: python scripts/prompt_lengths.py <records_dir> <out.json>
"""

import collections
import json
import pathlib
import sys

import numpy as np
import pi05_accel


def main():
    records, out = sys.argv[1], pathlib.Path(sys.argv[2])
    policy = pi05_accel.make_policy("stock", None)
    lengths, per_prompt = collections.Counter(), {}
    for p in sorted(pathlib.Path(records).glob("step_*.npy")):
        rec = np.load(p, allow_pickle=True).item()
        obs = {k.removeprefix("inputs/"): v for k, v in rec.items() if k.startswith("inputs/")}
        n = int(policy._input_transform(obs)["tokenized_prompt_mask"].sum())  # noqa: SLF001
        lengths[n] += 1
        per_prompt[str(obs["prompt"])] = n
    result = {
        "provenance": pi05_accel.provenance(__file__),
        "observations": sum(lengths.values()),
        "distinct_prompts": len(per_prompt),
        "valid_tokens_min": min(lengths),
        "valid_tokens_max": max(lengths),
        "histogram": dict(sorted(lengths.items())),
        "per_prompt": dict(sorted(per_prompt.items(), key=lambda kv: kv[1])),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print({k: result[k] for k in ("observations", "distinct_prompts", "valid_tokens_min", "valid_tokens_max")})


if __name__ == "__main__":
    main()
