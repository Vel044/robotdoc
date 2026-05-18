#!/usr/bin/env python3
"""
async_inference_timeline.py

生成三张 async inference 时间线图（情况 A/B/C 各一张）。

运行:
    conda activate lerobot
    python async_inference_timeline.py
输出:
    async_inference_A.png
    async_inference_B.png
    async_inference_C.png
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

plt.rcParams['font.sans-serif'] = [
    'PingFang SC', 'Noto Sans CJK SC', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))

# ── 配色 ─────────────────────────────────────────────────────────
C_EXEC  = '#2563EB'
C_STALL = '#CBD5E1'
C_DROP  = '#EF4444'
C_NEW   = '#22C55E'
C_ENS   = '#F59E0B'
C_INF   = '#7C3AED'


# ── 工具函数 ──────────────────────────────────────────────────────

FS_TITLE  = 25    # 图标题
FS_LABEL  = 22    # y 轴行标签
FS_BAR    = 18    # 色块内文字
FS_TRIG   = 16    # 触发/到货标注
FS_TICK   = 15    # 轴刻度
FS_LEGEND = 16    # 图例
FS_LANE   = 14    # 泳道左侧标签

BAR_Y = 0.30      # 色块起始 y（越大越居中）
BAR_H = 0.40      # 色块高度（减小=条更细）


def hbar(ax, x, w, color, xmax, y=BAR_Y, h=BAR_H,
         label=None, lc='white', fs=FS_BAR, zorder=2):
    if x >= xmax or w <= 0:
        return
    w = min(w, xmax - x)
    ax.broken_barh([(x, w)], (y, h), facecolors=color,
                   edgecolor='white', linewidth=0.5, zorder=zorder)
    if label and w > 12:
        ax.text(x + w / 2, y + h / 2, label,
                ha='center', va='center', fontsize=fs,
                color=lc, fontweight='bold', zorder=zorder + 1)


def fmt(ax, xmax, ylabel, tick=10, xticks=False, fs_label=FS_LABEL):
    ax.set_xlim(-2, xmax)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel(ylabel, fontsize=fs_label, rotation=0,
                  ha='right', va='center', labelpad=68)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    tks = list(range(0, xmax, tick))
    ax.set_xticks(tks)
    if xticks:
        ax.set_xticklabels([str(t) for t in tks], fontsize=FS_TICK, rotation=55, ha='right')
    else:
        ax.set_xticklabels([])
    for x in range(0, xmax, tick):
        lw = 0.7 if x % 50 == 0 else 0.2
        ax.axvline(x, color='#94A3B8', linewidth=lw, zorder=0)


def trig(ax, x, label, fs=FS_TRIG):
    ax.axvline(x, color='#1D4ED8', linewidth=1.3, linestyle='--', alpha=0.85, zorder=3)
    ax.text(x, 0.97, label, ha='center', va='top',
            fontsize=fs, color='#1D4ED8', fontweight='bold')


def arrv(ax, x, label, fs=FS_TRIG - 1):
    ax.axvline(x, color='#15803D', linewidth=1.1, linestyle='-.', alpha=0.75, zorder=3)
    ax.text(x, 0.06, label, ha='center', va='bottom', fontsize=fs, color='#15803D')


def draw_chunk_lanes(ax, xmax, lanes, stall_spans=None, tick=10,
                     fs_lane=FS_LANE, fs_seg=FS_BAR, show_xticks=False):
    """
    绘制多条 chunk 泳道。每条 lane 独占一行，从上到下排列。
    lanes: [{'name': str, 'arrival': int|None, 'segs': [(x, w, color, label), ...]}]
    """
    n   = len(lanes)
    gap = 0.06
    lh  = min(BAR_H + 0.04, (1.0 - gap * (n + 1)) / n)

    ax.set_xlim(-2, xmax)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.set_xticks(range(0, xmax, tick))
    if show_xticks:
        ax.set_xticklabels([str(x) for x in range(0, xmax, tick)],
                           fontsize=FS_TICK, rotation=0, ha='center')
        ax.tick_params(axis='x', pad=2)
    else:
        ax.set_xticklabels([])
    for x in range(0, xmax, tick):
        ax.axvline(x, color='#94A3B8',
                   linewidth=(0.7 if x % 50 == 0 else 0.2), zorder=0)

    if stall_spans:
        for sx, sw in stall_spans:
            sw = min(sw, xmax - sx)
            if sw > 0:
                ax.axvspan(sx, sx + sw, alpha=0.18, color='#6B7280', zorder=1)

    for i, lane in enumerate(lanes):
        y = 1.0 - (i + 1) * (lh + gap)

        # 泳道底色
        ax.axhspan(y, y + lh, alpha=0.06,
                   color='#1E3A5F' if i % 2 == 0 else '#3D1A1A', zorder=1)

        # 左侧标签
        tag = lane['name']
        if lane.get('arrival') is not None:
            tag += f"\n到货 f{lane['arrival']}"
        ax.text(-1.5, y + lh / 2, tag, ha='right', va='center',
                fontsize=fs_lane, color='#1E293B', fontweight='bold',
                transform=ax.transData)

        # 色块
        for sx, sw, c, lbl in lane['segs']:
            if sx >= xmax or sw <= 0:
                continue
            sw_clip = min(sw, xmax - sx)
            ax.broken_barh([(sx, sw_clip)], (y, lh),
                           facecolors=c, edgecolor='white', linewidth=0.5, zorder=2)
            if lbl and sw_clip > 16:
                ax.text(sx + sw_clip / 2, y + lh / 2, lbl,
                        ha='center', va='center', fontsize=fs_seg,
                        color='white', fontweight='bold', zorder=3)

        if i < n - 1:
            ax.axhline(y, color='#E2E8F0', linewidth=0.6, zorder=1)



def add_legend(fig):
    elems = [
        mpatches.Patch(color=C_EXEC,  label='正常执行'),
        mpatches.Patch(color=C_STALL, label='Stall（队列空，等推理）'),
        mpatches.Patch(color=C_INF,   label='推理进程（后台）'),
        mpatches.Patch(color=C_DROP,  label='Chunk 丢弃帧（已过期）'),
        mpatches.Patch(color=C_NEW,   label='Chunk 有效新帧'),
        mpatches.Patch(color=C_ENS,   label='Temporal Ensemble（融合区间）'),
    ]
    # 图例放在图的正下方，不与标题冲突
    fig.legend(handles=elems, loc='lower center', fontsize=FS_LEGEND,
               bbox_to_anchor=(0.55, 0.0), framealpha=0.93,
               ncol=6, edgecolor='#CBD5E1')


# ╔════════════════════════════════════════════════════════════════╗
# ║  图 A  T_infer=30帧  永不 stall，有 Temporal Ensemble        ║
# ╚════════════════════════════════════════════════════════════════╝

def plot_A():
    XMAX = 205
    # 行: exec / infer / chunk(4泳道，底部显示刻度)
    HR = [1.1, 0.55, 2.2]
    fig = plt.figure(figsize=(24, 12))
    gs  = GridSpec(len(HR), 1, height_ratios=HR,
                   hspace=0.07, left=0.13, right=0.97, top=0.90, bottom=0.13)
    ax  = [fig.add_subplot(gs[i]) for i in range(len(HR))]

    # ② 执行行：蓝色正常执行 + 黄色 Ensemble 区间，同一高度分段拼接，无叠加
    for x, w, c, lbl in [
        (0,   80, C_EXEC, '持续执行'),
        (80,  20, C_ENS,  'Ensemble\n[80..99]'),
        (100, 30, C_EXEC, ''),
        (130, 20, C_ENS,  'Ensemble\n[130..149]'),
        (150, 30, C_EXEC, ''),
        (180, 20, C_ENS,  'Ensemble\n[180..199]'),
        (200,  5, C_EXEC, ''),
    ]:
        hbar(ax[0], x, w, c, XMAX,
             label=lbl or None,
             lc='#78350F' if c == C_ENS else 'white')
    fmt(ax[0], XMAX, '② 执行', fs_label=FS_LABEL)
    ax[0].set_title(
        '情况 A：T_infer = 30帧（1 s）    [OK] 永不 stall，有 Temporal Ensemble    FPS = 30\n'
        'N=100   threshold=0.5（触发阈值=50帧）   F=30 FPS',
        fontsize=FS_TITLE, fontweight='bold', color='#166534', loc='left', pad=5)
    for x, lbl in [(50,'触发P1'), (100,'触发P2'), (150,'触发P3'), (200,'触发P4')]:
        trig(ax[0], x, lbl)

    # ③ 推理行
    fmt(ax[1], XMAX, '③ 推理进程', fs_label=FS_LABEL)
    for s, d, lbl in [(50,30,'P1  f50→f80'), (100,30,'P2  f100→f130'),
                      (150,30,'P3  f150→f180'), (200,30,'P4  f200→')]:
        hbar(ax[1], s, min(d, XMAX-s), C_INF, XMAX, label=lbl)
        if s + d <= XMAX:
            arrv(ax[1], s + d, f'f{s+d}')

    # ④ Chunk 泳道（每条 chunk 一行）
    ax[2].set_ylabel('④ Chunk内容\n(每行=一次到货)', fontsize=FS_LABEL,
                     rotation=0, ha='right', va='center', labelpad=68)
    draw_chunk_lanes(ax[2], XMAX, show_xticks=True, lanes=[
        {'name': 'A（初始）', 'arrival': 0,
         'segs': [(0,100,C_NEW,'A  [0..99]  全部有效')]},
        {'name': 'B', 'arrival': 80,
         'segs': [(50,30,C_DROP,'丢弃 [50..79]'),
                  (80,20,C_ENS,'Ens [80..99]'),
                  (100,50,C_NEW,'B 新增  [100..149]')]},
        {'name': 'C', 'arrival': 130,
         'segs': [(100,30,C_DROP,'丢弃 [100..129]'),
                  (130,20,C_ENS,'Ens [130..149]'),
                  (150,50,C_NEW,'C 新增  [150..199]')]},
        {'name': 'D', 'arrival': 180,
         'segs': [(150,30,C_DROP,'丢弃 [150..179]'),
                  (180,20,C_ENS,'Ens [180..199]'),
                  (200,50,C_NEW,'D→')]},
    ])
    add_legend(fig)
    path = os.path.join(HERE, 'async_inference_A.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'✅ A: {path}')
    plt.close(fig)


# ╔════════════════════════════════════════════════════════════════╗
# ║  图 B  T_infer=50帧  临界，无 Ensemble                       ║
# ╚════════════════════════════════════════════════════════════════╝

def plot_B():
    XMAX = 210
    HR = [1.1, 0.55, 2.0]
    fig = plt.figure(figsize=(24, 11))
    gs  = GridSpec(len(HR), 1, height_ratios=HR,
                   hspace=0.07, left=0.13, right=0.97, top=0.90, bottom=0.13)
    ax  = [fig.add_subplot(gs[i]) for i in range(len(HR))]

    hbar(ax[0], 0, 200, C_EXEC, XMAX, label='全程正常执行（临界，无 Ensemble）')
    fmt(ax[0], XMAX, '② 执行', fs_label=FS_LABEL)
    ax[0].set_title(
        '情况 B：T_infer = 50帧（1.67 s）    [临界] 无 Ensemble，偶发1帧 stall    FPS ≈ 30\n'
        'N=100   threshold=0.5   F=30 FPS',
        fontsize=FS_TITLE, fontweight='bold', color='#854D0E', loc='left', pad=5)
    for x, lbl in [(50,'触发P1'), (100,'触发P2\n(立即)'), (150,'触发P3\n(立即)'), (200,'触发P4\n(立即)')]:
        trig(ax[0], x, lbl)

    fmt(ax[1], XMAX, '③ 推理进程', fs_label=FS_LABEL)
    for s, d, lbl in [(50,50,'P1  f50→f100'), (100,50,'P2  f100→f150'),
                      (150,50,'P3  f150→f200'), (200,50,'P4  f200→')]:
        hbar(ax[1], s, min(d, XMAX-s), C_INF, XMAX, label=lbl)
        if s + d <= XMAX:
            arrv(ax[1], s + d, f'f{s+d}')

    ax[2].set_ylabel('④ Chunk内容\n(每行=一次到货)', fontsize=FS_LABEL,
                     rotation=0, ha='right', va='center', labelpad=68)
    draw_chunk_lanes(ax[2], XMAX, show_xticks=True, lanes=[
        {'name': 'A（初始）', 'arrival': 0,
         'segs': [(0,100,C_NEW,'A  [0..99]  全部有效')]},
        {'name': 'B', 'arrival': 100,
         'segs': [(50,50,C_DROP,'B 丢弃  [50..99]'),
                  (100,50,C_NEW,'B 新增  [100..149]')]},
        {'name': 'C', 'arrival': 150,
         'segs': [(100,50,C_DROP,'C 丢弃  [100..149]'),
                  (150,50,C_NEW,'C 新增  [150..199]')]},
        {'name': 'D', 'arrival': 200,
         'segs': [(150,50,C_DROP,'D 丢弃  [150..199]'),
                  (200,50,C_NEW,'D→')]},
    ])
    add_legend(fig)
    path = os.path.join(HERE, 'async_inference_B.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'✅ B: {path}')
    plt.close(fig)


# ╔════════════════════════════════════════════════════════════════╗
# ║  图 C  T_infer=70帧  周期性 stall                            ║
# ╚════════════════════════════════════════════════════════════════╝
#
# 关键：chunk 泳道用"实际执行帧位置"标注，而非抽象 timestep：
#   - Chunk B 到货 f120，50 帧有效动作在 f120-f170 真实执行  → 绿色块 [120..170]
#   - Chunk B 丢弃的是 f50-f100（chunk A 执行期）           → 红色块 [50..100]
#   - Chunk C 到货 f190，丢弃 = B 已执行的 [120..170]       → 红色块 [120..170]
#   - Chunk C 新增 = f190-f240 真实执行                     → 绿色块 [190..240]
#   - stall 期间（100-120, 170-190, 240+）显示灰底，不计过期

def plot_C():
    XMAX = 252
    HR = [1.1, 0.55, 1.8]
    fig = plt.figure(figsize=(24, 11))
    gs  = GridSpec(len(HR), 1, height_ratios=HR,
                   hspace=0.07, left=0.13, right=0.97, top=0.90, bottom=0.13)
    ax  = [fig.add_subplot(gs[i]) for i in range(len(HR))]

    for seg in [
        (0,  100, C_EXEC,  '执行 chunk A（100帧）'),
        (100, 20, C_STALL, 'stall\n20帧'),
        (120, 50, C_EXEC,  '执行 chunk B（50帧）'),
        (170, 20, C_STALL, 'stall\n20帧'),
        (190, 50, C_EXEC,  '执行 chunk C（50帧）'),
        (240, 12, C_STALL, 'stall…'),
    ]:
        hbar(ax[0], seg[0], seg[1], seg[2], XMAX,
             label=seg[3], lc='white' if seg[2] != C_STALL else '#374151')
    fmt(ax[0], XMAX, '② 执行', fs_label=FS_LABEL)
    ax[0].set_title(
        '情况 C：T_infer = 70帧（2.33 s）    [stall] 周期性 stall 20帧/周期    FPS ≈ 21.4\n'
        'N=100   threshold=0.5   F=30 FPS        稳态：执行50帧 → stall 20帧（周期=70帧）',
        fontsize=FS_TITLE, fontweight='bold', color='#BE123C', loc='left', pad=5)
    for x, lbl in [(50,'触发P1'), (120,'触发P2\n(立即)'), (190,'触发P3\n(立即)')]:
        trig(ax[0], x, lbl)

    fmt(ax[1], XMAX, '③ 推理进程', fs_label=FS_LABEL)
    for s, d, lbl in [(50,70,'P1  f50→f120'), (120,70,'P2  f120→f190'), (190,60,'P3  f190→f260→')]:
        hbar(ax[1], s, min(d, XMAX-s), C_INF, XMAX, label=lbl)
        if s + d <= XMAX:
            arrv(ax[1], s + d, f'f{s+d}')

    ax[2].set_ylabel('④ Chunk内容\n(坐标=真实执行帧)', fontsize=FS_LABEL,
                     rotation=0, ha='right', va='center', labelpad=68)
    draw_chunk_lanes(ax[2], XMAX,
        lanes=[
            {'name': 'A（初始）', 'arrival': 0,
             'segs': [(0, 100, C_NEW, 'A  [0..100]  全部有效')]},
            {'name': 'B', 'arrival': 120,
             'segs': [(50, 50, C_DROP, 'B 丢弃 [50..100]\n(chunk A 执行期)'),
                      (120, 50, C_NEW,  'B 新增  实际执行 [120..170]')]},
            {'name': 'C', 'arrival': 190,
             'segs': [(120, 50, C_DROP, 'C 丢弃 [120..170]\n(B 已执行)'),
                      (190, 50, C_NEW,  'C 新增  实际执行 [190..240]')]},
        ],
        stall_spans=[(100,20),(170,20),(240,12)],
        show_xticks=True,
    )
    for sx, lbl in [(105,'stall\n(不过期)'), (175,'stall\n(不过期)'), (244,'stall…')]:
        if sx < XMAX:
            ax[2].text(sx, 0.96, lbl, ha='center', va='top',
                       fontsize=FS_TRIG - 1, color='#6B7280', style='italic')
    add_legend(fig)
    path = os.path.join(HERE, 'async_inference_C.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'✅ C: {path}')
    plt.close(fig)


# ── 入口 ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    plot_A()
    plot_B()
    plot_C()
    print('全部完成。')
