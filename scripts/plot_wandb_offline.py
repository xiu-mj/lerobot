"""
解析 wandb 离线输出并可视化（从 output.log 或 wandb-history.jsonl / wandb-summary.json）。
用法示例：
python scripts/plot_wandb_offline.py --log "D:\\ubuntu2window\\wandb_offline\\wandb\\offline-run-20260711_142759-w9ccgblt\\files\\output.log" --out ./plots/loss

依赖：pandas matplotlib plotly
pip install pandas matplotlib plotly
"""
import re
import json
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--log", required=False, help="path to output.log")
parser.add_argument("--summary", required=False, help="path to wandb-summary.json")
parser.add_argument("--out", default="./plots/loss", help="output prefix (without ext)")
parser.add_argument("--metric", default="loss", help="metric key to extract from log lines (default: loss)")
args = parser.parse_args()

Path(args.out).parent.mkdir(parents=True, exist_ok=True)

# try parse output.log
df = None
if args.log:
    p = Path(args.log)
    if not p.exists():
        raise SystemExit(f"log not found: {p}")
    steps = []
    values = []
    xs = []
    # regex: capture step and metric (e.g. loss:0.392 or loss:1.33)
    step_re = re.compile(r"step[:=]\s*(\d+)")
    metric_re = re.compile(r"" + re.escape(args.metric) + r"[:=]([0-9eE+\-\.]+)")
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = step_re.search(line)
            m = metric_re.search(line)
            if s and m:
                try:
                    step = int(s.group(1))
                    val = float(m.group(1))
                except Exception:
                    continue
                steps.append(step)
                values.append(val)
    if steps:
        df = pd.DataFrame({"step": steps, args.metric: values})
        df = df.sort_values("step").drop_duplicates("step")

# try summary json
summary = None
if args.summary:
    p = Path(args.summary)
    if p.exists():
        try:
            summary = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            summary = None

if df is None and summary is None:
    raise SystemExit("没有在 log 或 summary 中找到可用指标。")

# plot with matplotlib
out_png = args.out + ".png"
plt.figure(figsize=(10,5))
if df is not None:
    plt.plot(df['step'], df[args.metric], label=args.metric)
if summary and args.metric in summary:
    # plot final point
    t = summary.get('_step', None)
    v = summary.get(args.metric, None)
    if v is not None and t is not None:
        plt.scatter([t], [v], color='red', label='summary')
plt.xlabel('step')
plt.ylabel(args.metric)
plt.title(f'{args.metric} over steps')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(out_png)
print(f"Saved PNG: {out_png}")

# also save CSV for inspection
if df is not None:
    out_csv = args.out + ".csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved CSV: {out_csv}")

print("Done.")
