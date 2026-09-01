from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from .run import MODE_ORDER, ROOT, load_config


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


METRICS = [
    "prompt_tokens",
    "completion_tokens",
    "local_context_total",
    "model_wall_seconds",
    "task_wall_seconds",
    "tool_action_count",
    "repeated_tool_action_count",
    "retry_count",
    "request_bytes",
    "response_bytes",
    "cpu_seconds",
    "rss_peak_bytes",
    "gpu_running_seconds",
    "gpu_dedicated_peak_bytes",
    "cost_cny",
]


def bootstrap_mean(values: np.ndarray, resamples: int, seed: int) -> tuple[float, float]:
    if not len(values):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    chunk = 1000
    for start in range(0, resamples, chunk):
        size = min(chunk, resamples - start)
        samples = rng.choice(values, size=(size, len(values)), replace=True)
        means[start : start + size] = samples.mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    z = float(norm.ppf(0.975))
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - radius, center + radius


def load_records(results_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted((results_dir / "runs").glob("main-*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        metrics = record["metrics"]
        rows.append(
            {
                "run_id": record["run_id"],
                "block": record["block"],
                "position": record["position"],
                "mode": record["mode"],
                "completed": record["quality"]["completed"],
                "task_success": record["quality"]["task_success"],
                "outcome": record["quality"]["outcome"],
                "iterations": len(record["api_turns"]),
                "local_context_total": metrics["local_context_tokens"]["local_total"],
                **{key: metrics.get(key) for key in METRICS if key != "local_context_total"},
            }
        )
    return pd.DataFrame(rows)


def aggregate(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for mode_index, mode in enumerate(item.value for item in MODE_ORDER):
        group = frame[frame["mode"] == mode]
        successes = int(group["task_success"].sum())
        success_low, success_high = wilson(successes, len(group))
        base = {
            "mode": mode,
            "n": len(group),
            "successes": successes,
            "success_rate": successes / len(group) if len(group) else np.nan,
            "success_rate_ci_low": success_low,
            "success_rate_ci_high": success_high,
            "success_cost_cny": group["cost_cny"].sum() / successes if successes else np.nan,
        }
        for metric_index, metric in enumerate(METRICS + ["iterations"]):
            values = group[metric].dropna().astype(float).to_numpy()
            low, high = bootstrap_mean(
                values,
                int(config["bootstrap_resamples"]),
                int(config["bootstrap_seed"]) + mode_index * 100 + metric_index,
            )
            base[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            base[f"{metric}_median"] = float(np.median(values)) if len(values) else np.nan
            base[f"{metric}_p95"] = float(np.quantile(values, 0.95)) if len(values) else np.nan
            base[f"{metric}_mean_ci_low"] = low
            base[f"{metric}_mean_ci_high"] = high
        rows.append(base)
    return pd.DataFrame(rows)


def paired_differences(frame: pd.DataFrame) -> pd.DataFrame:
    selected = ["task_success", "local_context_total", "task_wall_seconds", "cost_cny", "tool_action_count"]
    wide = frame.pivot(index="block", columns="mode", values=selected)
    rows = []
    for mode in (item.value for item in MODE_ORDER[1:]):
        for metric in selected:
            delta = wide[(metric, mode)] - wide[(metric, "full")]
            for block, value in delta.items():
                rows.append({"block": block, "mode": mode, "metric": metric, "delta_vs_full": value})
    return pd.DataFrame(rows)


def plot_results(frame: pd.DataFrame, summary: pd.DataFrame, records: list[dict[str, Any]], figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    order = [item.value for item in MODE_ORDER]
    labels = ["完整", "无历史", "无推理", "无工具定义", "无工具结果"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric, title, unit in zip(
        axes.ravel(),
        ["local_context_total", "task_wall_seconds", "cost_cny", "tool_action_count"],
        ["累计本地上下文令牌", "任务时延", "调用费用", "工具动作数"],
        ["令牌", "秒", "元", "次"],
    ):
        ax.boxplot([frame.loc[frame["mode"] == mode, metric] for mode in order], tick_labels=labels)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figures / "resource_distributions.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(order))
    rates = summary["success_rate"].to_numpy()
    errors = np.maximum(
        0.0,
        np.vstack((rates - summary["success_rate_ci_low"], summary["success_rate_ci_high"] - rates)),
    )
    ax.bar(x, rates, yerr=errors, capsize=5)
    ax.set_xticks(x, labels, rotation=20)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("成功率")
    ax.set_title("任务成功率与 95% 威尔逊置信区间")
    fig.tight_layout()
    fig.savefig(figures / "success_rate.png", dpi=220)
    plt.close(fig)

    growth_rows = []
    for record in records:
        for turn in record["api_turns"]:
            growth_rows.append(
                {
                    "mode": record["mode"],
                    "iteration": turn["iteration"],
                    "tokens": turn["context_tokens"]["local_total"],
                }
            )
    growth = pd.DataFrame(growth_rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    for mode, label in zip(order, labels):
        values = growth[growth["mode"] == mode].groupby("iteration")["tokens"].mean()
        ax.plot(values.index, values.values, marker="o", label=label)
    ax.set_xlabel("循环轮次")
    ax.set_ylabel("本轮上下文令牌")
    ax.set_title("上下文增长曲线")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "context_growth.png", dpi=220)
    plt.close(fig)


def report(
    summary: pd.DataFrame,
    differences: pd.DataFrame,
    results_dir: Path,
    config: dict[str, Any],
) -> None:
    display = summary[[
        "mode", "n", "success_rate", "success_rate_ci_low", "success_rate_ci_high",
        "local_context_total_mean", "local_context_total_median", "local_context_total_p95",
        "task_wall_seconds_mean", "task_wall_seconds_median", "task_wall_seconds_p95",
        "cost_cny_mean", "cost_cny_median", "cost_cny_p95", "success_cost_cny",
    ]].copy()
    for column in display.columns[2:]:
        display[column] = display[column].map(lambda value: "—" if pd.isna(value) else f"{value:.6f}")
    resource_display = summary[[
        "mode", "cpu_seconds_mean", "rss_peak_bytes_mean", "gpu_running_seconds_mean",
        "gpu_dedicated_peak_bytes_mean", "request_bytes_mean", "response_bytes_mean", "retry_count_mean",
    ]].copy()
    resource_display["rss_peak_mib_mean"] = resource_display.pop("rss_peak_bytes_mean") / 1024**2
    resource_display["gpu_dedicated_peak_mib_mean"] = resource_display.pop("gpu_dedicated_peak_bytes_mean") / 1024**2
    for column in resource_display.columns[1:]:
        resource_display[column] = resource_display[column].map(lambda value: f"{value:.6f}")
    delta_display = (
        differences.groupby(["mode", "metric"], as_index=False)["delta_vs_full"]
        .mean()
        .pivot(index="mode", columns="metric", values="delta_vs_full")
        .reset_index()
    )
    for column in delta_display.columns[1:]:
        delta_display[column] = delta_display[column].map(lambda value: f"{value:.6f}")
    total_cost = float((summary["cost_cny_mean"] * summary["n"]).sum())
    text = f"""# Kimi K3 智能体上下文资源消融实验报告

## 实验设置

- 任务：实验 1-1 受约束多币种计算任务
- 模型：`{config['model']}`
- 样本：五种配置各 {config['repetitions']} 次，共 {int(config['repetitions']) * 5} 次
- 调度：五阶循环拉丁方，每种配置在每个顺序位置出现 6 次
- 置信区间：连续指标采用 {config['bootstrap_resamples']:,} 次分块自助法；成功率采用威尔逊区间
- 价格：缓存输入 {config['price_cny_per_million_tokens']['cached_input']} 元、非缓存输入 {config['price_cny_per_million_tokens']['uncached_input']} 元、输出 {config['price_cny_per_million_tokens']['output']} 元/百万令牌

## 配置级结果

{display.to_markdown(index=False)}

字段定义：`local_context_total` 为任务各轮本地分词上下文之和；`success_cost_cny` 为该配置总费用除以成功次数。正式实验服务商计费总额为 {total_cost:.6f} 元。

## 相对完整配置的区组内差值

{delta_display.to_markdown(index=False)}

差值按同一拉丁方区组内的消融配置减去完整配置计算，再对 30 个区组取均值。

## 本地与传输资源

{resource_display.to_markdown(index=False)}

## 可复算公式

设服务商报告的缓存输入、非缓存输入和输出令牌量分别为 $N_c$、$N_u$、$N_o$，对应单价为 $P_c$、$P_u$、$P_o$，则单任务费用为：

$$
C = \frac{{N_cP_c + N_uP_u + N_oP_o}}{{10^6}}
$$

任务时延由模型调用、工具执行、验证和框架开销组成：

$$
T_{{task}} = T_{{model}} + T_{{tool}} + T_{{verify}} + T_{{framework}}
$$

每个任务文件均保存上述分量，验收程序从原始字段重新计算费用并检查时延闭合。

## 可复算产物

- `events.jsonl`：请求、响应、工具、验证和任务事件
- `runs/`：150 个任务级完整记录
- `derived/run_metrics.csv` 与 `run_metrics.parquet`：任务级指标
- `derived/config_summary.csv` 与 `config_summary.parquet`：配置级统计
- `derived/paired_differences.csv`：同一拉丁方区组内相对完整配置的差值
"""
    (results_dir / "REPORT.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path)
    args = parser.parse_args()
    config = load_config()
    results_dir = args.results_dir or ROOT / "resource_experiment" / "results" / config["experiment_id"]
    records = load_records(results_dir)
    if not records:
        raise RuntimeError("未找到正式实验任务记录。")
    frame = to_frame(records)
    summary = aggregate(frame, config)
    differences = paired_differences(frame)
    derived = results_dir / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    for name, table in (("run_metrics", frame), ("config_summary", summary), ("paired_differences", differences)):
        table.to_csv(derived / f"{name}.csv", index=False, encoding="utf-8-sig")
        table.to_parquet(derived / f"{name}.parquet", index=False)
    plot_results(frame, summary, records, results_dir / "figures")
    report(summary, differences, results_dir, config)
    print(f"分析完成：{len(frame)} 个任务，{len(summary)} 个配置。")


if __name__ == "__main__":
    main()
