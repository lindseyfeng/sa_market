#!/usr/bin/env python3
"""Emit one line per completed run across all three queued experiments.

Also emits terminal states: a crash signature in any log, or every training
process gone while runs are still outstanding.  Silence must not be able to
look like progress.
"""
import json, os, re, subprocess, sys, time

# spatial runs first now
PHASES = [("spatial_2x2_results.json","spatial",12, "cfg"),
          ("stability_results.json", "zoo",    24, "arm"),
          ("dose_results.json",      "dose",   12, "arm")]
LOGS = ["spatial.log", "stability.log", "dose.log"]
BAD = re.compile(r"Traceback|Error|FAILED|Killed|MemoryError|No such file", re.I)

seen, log_pos, quiet = set(), {f: 0 for f in LOGS}, 0
print("watching: zoo 24 -> dose 12 -> spatial 12", flush=True)

while True:
    total_done = 0
    for path, name, n, key in PHASES:
        if not os.path.exists(path):
            continue
        try:
            rows = json.load(open(path))
        except Exception:
            continue
        total_done += len(rows)
        for r in rows:
            k = (name, r[key], r["seed"])
            if k in seen:
                continue
            seen.add(k)
            extra = ""
            if "test_mae_cherry" in r:
                extra = f"  cherry {r['test_mae_cherry']:.3f}"
            print(f"[{name} {len(rows)}/{n}] {r[key]} seed {r['seed']}: "
                  f"MAE {r['test_mae']:.3f}{extra}  (epoch {r['epoch']})", flush=True)

    for f in LOGS:
        if not os.path.exists(f):
            continue
        with open(f, errors="ignore") as fh:
            fh.seek(log_pos[f])
            for line in fh:
                if BAD.search(line):
                    print(f"!! {f}: {line.strip()[:160]}", flush=True)
            log_pos[f] = fh.tell()

    alive = subprocess.run(["pgrep", "-f", "run_three_arms.py|run_spatial_2x2.py"],
                           capture_output=True).returncode == 0
    if total_done >= 48:
        print(f"ALL DONE: {total_done} runs", flush=True)
        sys.exit(0)
    if not alive:
        quiet += 1
        if quiet >= 3:
            print(f"!! no training process alive, {total_done}/48 runs done "
                  f"-- the queue has stopped", flush=True)
            sys.exit(1)
    else:
        quiet = 0
    time.sleep(90)
