import csv
from collections import defaultdict

DATA_PATH = '../timing_stats.csv'

TARGETS = {
    'so101_act_bottle_cs100': 'bottle_pick (cs100)',
    'so101_act_bottle_push': 'bottle_push (cs100默认)',
    'so101_act_bottle_classification': 'classification (cs100默认)',
}

PCT_COLS = ['obs_pct', 'inference_pct', 'action_pct', 'wait_pct']

rows = []
with open(DATA_PATH) as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

stats = defaultdict(lambda: defaultdict(list))
for row in rows:
    model = row['model']
    if model in TARGETS:
        for col in PCT_COLS:
            stats[model][col].append(float(row[col]))

for model, label in TARGETS.items():
    d = stats[model]
    n = len(d['obs_pct'])
    print(f"\n### {label} (n={n})")
    for col in PCT_COLS:
        vals = d[col]
        avg = sum(vals) / len(vals)
        mn, mx = min(vals), max(vals)
        print(f"  {col}: avg={avg:.1f}%  min={mn:.1f}%  max={mx:.1f}%")
