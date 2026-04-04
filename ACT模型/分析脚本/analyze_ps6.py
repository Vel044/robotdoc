#!/usr/bin/env python3
import re
import sys
from datetime import datetime

def parse_ps6_log(filename):
    """解析ps6日志文件，提取关键数据"""
    total_context = 0
    total_sleep = 0
    total_cpu = 0
    call_count = 0
    python_calls = []

    with open(filename, 'r') as f:
        for line in f:
            # 匹配 python-37210 的 pselect6 调用
            # 格式: python-37210 [002] ..... 3603.628701: do_select: pselect6: total=83464 ns, sleep=77574 ns, context=5890 ns, ret=1, end_time=3603.618293643 s
            match = re.search(r'python-(\d+).*pselect6:.*total=(\d+) ns,.*sleep=(\d+) ns,.*context=(\d+) ns', line)
            if match:
                pid = match.group(1)
                total = int(match.group(2))
                sleep = int(match.group(3))
                context = int(match.group(4))
                cpu = total - sleep  # cpu时间 = 总时间 - 睡眠时间

                call_count += 1
                total_context += context
                total_sleep += sleep
                total_cpu += cpu

                python_calls.append({
                    'pid': pid,
                    'total': total,
                    'sleep': sleep,
                    'context': context,
                    'cpu': cpu
                })

    return {
        'call_count': call_count,
        'total_context': total_context,
        'total_sleep': total_sleep,
        'total_cpu': total_cpu,
        'calls': python_calls
    }

def analyze_robot_action_time(filename):
    """分析机器人动作时间 - 通过end_time计算"""
    # 从end_time计算总的机器人动作时间
    end_times = []
    with open(filename, 'r') as f:
        for line in f:
            match = re.search(r'pselect6:.*end_time=([\d.]+) s', line)
            if match:
                end_time = float(match.group(1))
                end_times.append(end_time)

    if len(end_times) >= 2:
        robot_action_time = end_times[-1] - end_times[0]
        return robot_action_time * 1e9  # 转换为纳秒
    return 0

def main():
    if len(sys.argv) < 2:
        print("用法: python3 analyze_ps6.py <日志文件>")
        print("示例: python3 analyze_ps6.py /tmp/ps6_log.txt")
        sys.exit(1)

    filename = sys.argv[1]

    print(f"分析文件: {filename}")
    print("=" * 50)

    # 解析日志
    stats = parse_ps6_log(filename)

    print(f"\n【Python pselect6 调用统计】")
    print(f"  调用次数: {stats['call_count']}")
    print(f"  总context时间: {stats['total_context']:,} ns ({stats['total_context']/1e6:.2f} ms)")
    print(f"  总sleep时间: {stats['total_sleep']:,} ns ({stats['total_sleep']/1e6:.2f} ms)")
    print(f"  总CPU时间: {stats['total_cpu']:,} ns ({stats['total_cpu']/1e6:.2f} ms)")

    # 计算机器人动作时间
    robot_action_time = analyze_robot_action_time(filename)

    print(f"\n【时间占比分析】")
    print(f"  机器人总动作时间: {robot_action_time:,.0f} ns ({robot_action_time/1e6:.2f} ms)")

    if robot_action_time > 0:
        context_ratio = (stats['total_context'] / robot_action_time) * 100
        cpu_ratio = (stats['total_cpu'] / robot_action_time) * 100

        print(f"\n  ★ CPU时间占机器人动作时间比例: {cpu_ratio:.4f}%")
        print(f"  ★ Context时间占机器人动作时间比例: {context_ratio:.4f}%")

    # 详细分析
    print(f"\n【详细调用分析】")
    if stats['calls']:
        print(f"  最大context时间: {max(c['context'] for c in stats['calls']):,} ns")
        print(f"  最小context时间: {min(c['context'] for c in stats['calls']):,} ns")
        print(f"  平均context时间: {stats['total_context']//stats['call_count']:,} ns")

if __name__ == "__main__":
    main()
