#!/usr/bin/env python3
import re
import sys

def parse_futex_log(filename):
    """解析futex日志文件，提取关键数据"""
    total_cpu = 0
    total_sleep = 0
    call_count = 0

    process_stats = {}  # 按进程统计
    op_stats = {}       # 按操作类型统计

    with open(filename, 'r') as f:
        for line in f:
            # 匹配 futex 调用
            # 格式: python-37210 [002] ..... 3623.423764: __arm64_sys_futex: futex: op=9, val=0, ret=0, total=3737989 ns, cpu=481 ns, sleep=3737508 ns
            match = re.search(r'(\S+)-(\d+).*futex:.*op=(\d+),.*total=(\d+) ns,.*cpu=(\d+) ns,.*sleep=(\d+) ns', line)
            if match:
                process_name = match.group(1)
                pid = match.group(2)
                op = match.group(3)
                total = int(match.group(4))
                cpu = int(match.group(5))
                sleep = int(match.group(6))

                call_count += 1
                total_cpu += cpu
                total_sleep += sleep

                # 按进程统计
                key = f"{process_name}-{pid}"
                if key not in process_stats:
                    process_stats[key] = {'count': 0, 'cpu': 0, 'sleep': 0}
                process_stats[key]['count'] += 1
                process_stats[key]['cpu'] += cpu
                process_stats[key]['sleep'] += sleep

                # 按操作类型统计
                if op not in op_stats:
                    op_stats[op] = {'count': 0, 'cpu': 0, 'sleep': 0}
                op_stats[op]['count'] += 1
                op_stats[op]['cpu'] += cpu
                op_stats[op]['sleep'] += sleep

    return {
        'call_count': call_count,
        'total_cpu': total_cpu,
        'total_sleep': total_sleep,
        'process_stats': process_stats,
        'op_stats': op_stats
    }

def analyze_robot_action_time(filename):
    """分析机器人动作时间 - 通过时间戳计算"""
    timestamps = []
    with open(filename, 'r') as f:
        for line in f:
            # 匹配时间戳
            match = re.search(r'\[\d+\]\s+\.\.\.\.\.\s+([\d.]+):', line)
            if match:
                timestamp = float(match.group(1))
                timestamps.append(timestamp)

    if len(timestamps) >= 2:
        return (timestamps[-1] - timestamps[0]) * 1e9  # 转换为纳秒
    return 0

def main():
    if len(sys.argv) < 2:
        print("用法: python3 analyze_futex.py <日志文件>")
        print("示例: python3 analyze_futex.py /tmp/futex_log.txt")
        sys.exit(1)

    filename = sys.argv[1]

    print(f"分析文件: {filename}")
    print("=" * 50)

    # 解析日志
    stats = parse_futex_log(filename)

    print(f"\n【总体统计】")
    print(f"  总调用次数: {stats['call_count']:,}")
    print(f"  总CPU时间: {stats['total_cpu']:,} ns ({stats['total_cpu']/1e6:.2f} ms)")
    print(f"  总睡眠时间: {stats['total_sleep']:,} ns ({stats['total_sleep']/1e6:.2f} ms)")

    # 计算机器人动作时间
    robot_action_time = analyze_robot_action_time(filename)

    print(f"\n【时间占比分析】")
    print(f"  机器人总动作时间: {robot_action_time:,.0f} ns ({robot_action_time/1e6:.2f} ms)")

    if robot_action_time > 0:
        cpu_ratio = (stats['total_cpu'] / robot_action_time) * 100
        sleep_ratio = (stats['total_sleep'] / robot_action_time) * 100

        print(f"\n  ★ CPU时间占机器人动作时间比例: {cpu_ratio:.6f}%")
        print(f"  ★ 睡眠时间占机器人动作时间比例: {sleep_ratio:.4f}%")

    # 按进程统计
    print(f"\n【按进程统计 (Top 10)】")
    sorted_processes = sorted(stats['process_stats'].items(), key=lambda x: x[1]['cpu'], reverse=True)
    for i, (proc, data) in enumerate(sorted_processes[:10]):
        print(f"  {i+1}. {proc}: {data['count']}次, CPU={data['cpu']:,}ns, Sleep={data['sleep']:,}ns")

    # 按操作类型统计
    print(f"\n【按操作类型统计】")
    op_names = {
        '0': 'FUTEX_WAIT',
        '1': 'FUTEX_WAKE',
        '9': 'FUTEX_WAIT_BITSET'
    }
    for op, data in sorted(stats['op_stats'].items()):
        op_name = op_names.get(op, f'op={op}')
        print(f"  {op_name}: {data['count']}次, CPU={data['cpu']:,}ns")

if __name__ == "__main__":
    main()
