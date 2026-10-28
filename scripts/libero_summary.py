"""LIBERO success rates of the runs in results/libero/ (from the stock client's logs).

Usage: python scripts/libero_summary.py   ->  results/libero_summary.json
"""

import json
import pathlib

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def success(log):
    res = [line.strip().endswith("True") for line in log.read_text().splitlines() if "INFO:root:Success:" in line]
    assert len(res) == 500, (log, len(res))  # 10 tasks x 50 initial states
    return sum(res)


def main():
    out = {}
    for run in sorted(pathlib.Path("results/libero").iterdir()):
        per = {s: success(run / s / "client.log") for s in SUITES}
        out[run.name] = {**per, "total": sum(per.values()), "rate": sum(per.values()) / 2000}
        print(f"{run.name:20s} {out[run.name]['total']} / 2000  {100 * out[run.name]['rate']:.2f}%")
    pathlib.Path("results/libero_summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
