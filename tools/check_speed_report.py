import argparse
import csv
import json
import math
import statistics


def evaluate(rows):
    groups = {case: [r for r in rows if r["case"] == case] for case in ("A", "B", "E")}
    if any(len(group) < 3 for group in groups.values()):
        raise ValueError("at least three measured A, B and E trials required")
    if len({r["steps"] for group in groups.values() for r in group}) != 1:
        raise ValueError("compare one step count at a time")
    medians = {}
    generations = {}
    p95 = {}
    for case, group in groups.items():
        trial_ids = [r["trial"] for r in group]
        if len(set(trial_ids)) != len(trial_ids):
            raise ValueError("duplicate trial within case")
        times = [float(r["sampling_seconds"]) for r in group]
        full = [float(r["generation_seconds"]) for r in group]
        if any(not math.isfinite(v) or v <= 0 for v in times + full):
            raise ValueError("finite positive durations required")
        medians[case] = statistics.median(times)
        generations[case] = statistics.median(full)
        p95[case] = sorted(times)[math.ceil(.95 * len(times)) - 1]
    quality = all(r["quality_pass"] == "yes" for r in groups["E"])
    hits = all(int(r["cache_hits"]) > 0 for r in groups["E"])
    passed = (medians["E"] <= .9 * medians["B"]
              and medians["E"] <= medians["A"]
              and generations["E"] <= min(generations["A"], generations["B"])
              and p95["E"] <= 1.05 * min(p95["A"], p95["B"])
              and quality and hits)
    return dict(median_sampling_seconds=medians, median_generation_seconds=generations,
                p95_sampling_seconds=p95, candidate_quality_pass=quality,
                candidate_has_hits=hits, timing_quality_gate=passed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    args = parser.parse_args()
    with open(args.csv, newline="", encoding="utf-8") as stream:
        result = evaluate(list(csv.DictReader(stream)))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["timing_quality_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
