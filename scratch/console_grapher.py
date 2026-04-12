import httpx
import time
import sys
import re
import os

# ANSI Colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

def get_color(score):
    if score >= 0.8: return GREEN
    if score >= 0.5: return YELLOW
    return RED

def render_bar(score, width=20):
    filled = int(score * width)
    bar = "#" * filled + "-" * (width - filled)
    return bar

def fetch_metrics(url="http://localhost:8000/metrics"):
    try:
        resp = httpx.get(url, timeout=5.0)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"{RED}Error fetching metrics: {e}{RESET}")
        return None

def parse_metrics(text):
    # We want to extract task name and score from data_cleaner_task_score_sum / data_cleaner_task_score_count
    # data_cleaner_task_score_sum{task="field-extraction"} 0.78
    # data_cleaner_task_score_count{task="field-extraction"} 1.0
    
    scores = {}
    
    # Use regex to find task names and values
    pattern_sum = r'data_cleaner_task_score_sum\{task="([^"]+)"\} ([\d\.]+)'
    pattern_count = r'data_cleaner_task_score_count\{task="([^"]+)"\} ([\d\.]+)'
    
    sums = dict(re.findall(pattern_sum, text))
    counts = dict(re.findall(pattern_count, text))
    
    for task in sums:
        s = float(sums[task])
        c = float(counts.get(task, 1.0))
        if c > 0:
            scores[task] = s / c
            
    return scores

def display_graph(scores):
    if not scores:
        print(f"{YELLOW}No metrics found yet. Environment may still be warming up.{RESET}")
        return

    print(f"\n{BOLD}{CYAN}RL TRAINING PERFORMANCE SUMMARY{RESET}")
    print("=" * 70)
    print(f"{BOLD}{'TASK':<35} {'SCORE':<8} {'PROGRESSION'}{RESET}")
    print("-" * 70)

    all_scores = []
    for task, score in sorted(scores.items()):
        color = get_color(score)
        bar = render_bar(score)
        print(f"{task:<35} {color}{score:>5.2f}{RESET}   [{color}{bar}{RESET}] {int(score*100):>3}%")
        all_scores.append(score)

    avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    color_avg = get_color(avg)
    
    print("-" * 70)
    print(f"{BOLD}GLOBAL AVERAGE:{RESET} {color_avg}{avg:.4f}{RESET}")
    print("=" * 70)

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/metrics"
    metrics_text = fetch_metrics(url)
    if metrics_text:
        scores = parse_metrics(metrics_text)
        display_graph(scores)
