"""Serve a pi05_accel policy variant over openpi's websocket protocol.

Drop-in for openpi's scripts/serve_policy.py with the pi05_libero checkpoint, so
the stock LIBERO client (examples/libero/main.py) works unchanged.

The denoising noise is derived from the observation's content and --seed, so it does
not depend on request order across parallel clients or on inference speed, and
every variant gets the same noise for a bit-identical observation. In closed loop
that is only the first request of an episode: once two variants' actions differ at
all, later observations differ and their noise is effectively independent. One
warm-up inference runs before serving and its time is logged (compile vs cache hit).

Usage: python scripts/serve.py [--variant V] [--compile MODE|none] [--port P] [--record DIR] [--seed S]
  --record saves every request and response with openpi's PolicyRecorder.
"""

import argparse
import hashlib
import logging
import time

import numpy as np
from openpi.policies import libero_policy
from openpi.policies import policy as _policy
from openpi.serving import websocket_policy_server
import pi05_accel


class ObservationSeededNoise:
    """Wraps a policy: noise = f(seed, hash of the observation)."""

    def __init__(self, policy, seed):
        self._policy, self._seed = policy, seed
        cfg = policy._model.config  # noqa: SLF001
        self._shape = (cfg.action_horizon, cfg.action_dim)
        self.metadata = policy.metadata

    def infer(self, obs):
        h = hashlib.sha256()
        for k in sorted(obs):
            v = np.asarray(obs[k])
            if v.dtype == object:  # its bytes would be pointers, different in every process
                raise TypeError(f"cannot hash observation field {k!r} of object dtype")
            h.update(k.encode())
            h.update(np.ascontiguousarray(v).tobytes())
        rng = np.random.default_rng([self._seed, int.from_bytes(h.digest()[:8], "little")])
        return self._policy.infer(obs, noise=rng.standard_normal(self._shape).astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="stock", help='"stock" or "+"-joined pi05_accel.FEATURES')
    ap.add_argument("--compile", default="max-autotune", help='torch.compile mode, or "none" for eager')
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--record", default=None)
    ap.add_argument("--seed", type=int, default=0, help="seed for the observation-derived denoising noise")
    ap.add_argument("--num-steps", type=int, default=10, help="denoising steps (static variants; behaviour-changing)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)

    policy = pi05_accel.make_policy(
        args.variant, None if args.compile == "none" else args.compile, num_steps=args.num_steps
    )
    t0 = time.perf_counter()
    policy.infer(libero_policy.make_libero_example())  # compile now, not during the first episode
    logging.info("warm-up inference %.1f s", time.perf_counter() - t0)
    metadata = policy.metadata  # PolicyRecorder does not forward it
    policy = ObservationSeededNoise(policy, args.seed)
    if args.record:
        policy = _policy.PolicyRecorder(policy, args.record)
    logging.info("provenance %s", pi05_accel.provenance(__file__))
    logging.info("serving variant=%s compile=%s num_steps=%d seed=%d on port %d",
                 args.variant, args.compile, args.num_steps, args.seed, args.port)
    server = websocket_policy_server.WebsocketPolicyServer(policy, host="127.0.0.1", port=args.port, metadata=metadata)
    server.serve_forever()


if __name__ == "__main__":
    main()
