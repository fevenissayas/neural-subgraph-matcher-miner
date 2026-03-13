import subprocess, time, json, os

CONFIGS = [
    # (label,               strategy, n_trials, n_neighborhoods, max_pattern_size)
    ("greedy_small",        "greedy",  2,        50,              4),
    ("greedy_medium",       "greedy",  5,        100,             5),
    ("greedy_large",        "greedy",  10,       200,             5),
    ("mcts_small",          "mcts",    2,        50,              4),
    ("mcts_medium",         "mcts",    5,        100,             5),
    ("mcts_large",          "mcts",    10,       200,             5),
    ("beam_small",          "beam",    2,        50,              4),
    ("beam_medium",         "beam",    5,        100,             5),
    ("beam_large",          "beam",    10,       200,             5),
]

env = os.environ.copy()
torch_lib = os.path.join(os.getcwd(), ".venv/lib/python3.7/site-packages/torch/lib")
env["LD_LIBRARY_PATH"] = torch_lib + ":" + env.get("LD_LIBRARY_PATH", "")

results = []
for label, strategy, n_trials, n_neighborhoods, max_sz in CONFIGS:
    out_path = f"results/experiments/{label}.json"
    cmd = [
        "python3", "-m", "subgraph_mining.decoder",
        "--dataset", "data/wiki-vote.pkl",
        "--model_path", "ckpt/model.pt",
        "--search_strategy", strategy,
        "--n_trials", str(n_trials),
        "--n_neighborhoods", str(n_neighborhoods),
        "--max_pattern_size", str(max_sz),
        "--min_pattern_size", "3",
        "--out_path", out_path,
        "--out_batch_size", "3",
    ]
    print(f"[RUN] {label} ...", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = round(time.time() - t0, 2)

    # Count patterns from json output (decoder saves as a list of pattern dicts)
    n_patterns = 0
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                data = json.load(f)
            if isinstance(data, list):
                n_patterns = len(data)
            elif isinstance(data, dict):
                # nested under "patterns" or "motifs" key
                n_patterns = len(data.get("patterns", data.get("motifs", data.get("results", []))))
        except Exception:
            n_patterns = 0

    row = {
        "label": label,
        "strategy": strategy,
        "n_trials": n_trials,
        "n_neighborhoods": n_neighborhoods,
        "max_pattern_size": max_sz,
        "runtime_s": elapsed,
        "n_patterns": n_patterns,
        "exit_code": proc.returncode,
    }
    results.append(row)
    print(f"  -> runtime={elapsed}s  patterns={n_patterns}  exit={proc.returncode}", flush=True)

with open("results/experiments/all_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nDone. Results saved to results/experiments/all_results.json")
