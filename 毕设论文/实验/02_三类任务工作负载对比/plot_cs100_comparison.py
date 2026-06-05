import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = '../timing_stats.csv'

TARGETS = {
    'so101_act_bottle_cs100': 'pick',
    'so101_act_bottle_push': 'push',
    'so101_act_bottle_classification': 'classification',
}

PCT_COLS = ['obs_pct', 'inference_pct', 'action_pct', 'wait_pct']
PHASE_LABELS = ['obs', 'inference', 'action', 'wait']
# 三个任务各一色
TASK_COLORS = ['#4C72B0', '#DD8452', '#55A868']

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

tasks = list(TARGETS.keys())
task_names = [TARGETS[t] for t in tasks]

# avgs[phase][task_idx], stds 同理
avgs = [[sum(stats[t][col]) / len(stats[t][col]) for t in tasks] for col in PCT_COLS]
stds  = [[np.std(stats[t][col])                   for t in tasks] for col in PCT_COLS]

n_phases = len(PCT_COLS)
n_tasks  = len(tasks)
width = 0.22
x = np.arange(n_phases)  # x轴：四个阶段

fig, ax = plt.subplots(figsize=(10, 5))

for i, (task_name, color) in enumerate(zip(task_names, TASK_COLORS)):
    offset = (i - (n_tasks - 1) / 2) * width
    vals = [avgs[p][i] for p in range(n_phases)]
    errs = [stds[p][i]  for p in range(n_phases)]
    bars = ax.bar(x + offset, vals, width, yerr=errs,
                  label=task_name, color=color, capsize=4, alpha=0.88)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{v:.1f}%', ha='center', va='bottom', fontsize=8)

ax.set_ylabel('Time Percentage (%)')
ax.set_title('Time Phase Comparison across Tasks (cs100)')
ax.set_xticks(x)
ax.set_xticklabels(PHASE_LABELS, fontsize=13)
ax.set_ylim(0, 65)
ax.legend(loc='upper left')
ax.yaxis.grid(True, linestyle='--', alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('cs100_task_comparison.png', dpi=150)
print("saved: cs100_task_comparison.png")
plt.show()
