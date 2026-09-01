import json
from pathlib import Path

import matplotlib.pyplot as plt


# ==================== 可修改参数 ====================

RESULT_FILES = {
    "Run 1 - Before Fix":
        "results/benchmark_run1_before_reset_fix.json",

    "Main Benchmark":
        "results/benchmark_run2_main_60.json",

    "Complex Stress Test":
        "results/benchmark_complex_30.json",
}

# 哪一轮作为最终主 Benchmark 展示
MAIN_RESULT = "Main Benchmark"

OUTPUT_DIR = Path("results/figures")


# ==================== 读取结果 ====================

def load_result(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_all_results():
    results = {}

    for name, path in RESULT_FILES.items():
        results[name] = load_result(path)

    return results


# ==================== 工具函数 ====================

def add_value_labels(ax, values, percent=False):
    for index, value in enumerate(values):
        if percent:
            text = f"{value * 100:.1f}%"
        else:
            text = f"{value:.2f}"

        ax.text(
            index,
            value,
            text,
            ha="center",
            va="bottom",
        )


# ==================== 图 1：成功率 ====================

def plot_success_rates(result):
    summary = result["summary"]

    labels = [
        "Task",
        "Vision",
        "Pick",
        "Place",
        "Recovery",
    ]

    values = [
        summary["task_success_rate"],
        summary["vision_success_rate"],
        summary["pick_success_rate"],
        summary["place_success_rate"],
        summary["recovery_success_rate"],
    ]

    fig, ax = plt.subplots(
        figsize=(9, 5),
    )

    ax.bar(
        labels,
        values,
    )

    ax.set_title(
        "Main Benchmark Success Rates"
    )

    ax.set_ylabel(
        "Success Rate"
    )

    ax.set_ylim(
        0,
        1.1,
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    add_value_labels(
        ax,
        values,
        percent=True,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "01_success_rates.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ==================== 图 2：定位 / 放置误差 ====================

def plot_error_metrics(result):
    summary = result["summary"]

    # JSON 中单位为米，转成毫米更直观
    localization_error_mm = (
        summary["mean_localization_error"]
        * 1000
    )

    place_error_mm = (
        summary["mean_place_error"]
        * 1000
    )

    labels = [
        "Localization Error",
        "Place Error",
    ]

    values = [
        localization_error_mm,
        place_error_mm,
    ]

    fig, ax = plt.subplots(
        figsize=(7, 5),
    )

    ax.bar(
        labels,
        values,
    )

    ax.set_title(
        "Mean Spatial Errors"
    )

    ax.set_ylabel(
        "Error (mm)"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    add_value_labels(
        ax,
        values,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "02_error_metrics.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ==================== 图 3：多轮 Benchmark 对比 ====================

def plot_benchmark_comparison(results):
    names = list(
        results.keys()
    )

    task_rates = [
        results[name]["summary"][
            "task_success_rate"
        ]
        for name in names
    ]

    place_rates = [
        results[name]["summary"][
            "place_success_rate"
        ]
        for name in names
    ]

    x = list(
        range(len(names))
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(10, 6),
    )

    task_positions = [
        position - width / 2
        for position in x
    ]

    place_positions = [
        position + width / 2
        for position in x
    ]

    task_bars = ax.bar(
        task_positions,
        task_rates,
        width,
        label="Task Success",
    )

    place_bars = ax.bar(
        place_positions,
        place_rates,
        width,
        label="Place Success",
    )

    ax.set_title(
        "Benchmark Comparison"
    )

    ax.set_ylabel(
        "Success Rate"
    )

    ax.set_ylim(
        0,
        1.1,
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        names,
        rotation=10,
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    ax.bar_label(
        task_bars,
        labels=[
            f"{value * 100:.1f}%"
            for value in task_rates
        ],
        padding=3,
    )

    ax.bar_label(
        place_bars,
        labels=[
            f"{value * 100:.1f}%"
            for value in place_rates
        ],
        padding=3,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "03_benchmark_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ==================== 图 4：Planner 指标 ====================

def plot_planner_metrics(results):
    names = list(
        results.keys()
    )

    invalid_rates = [
        results[name]["summary"][
            "invalid_action_rate"
        ] * 100
        for name in names
    ]

    planner_steps = [
        results[name]["summary"][
            "average_planner_steps"
        ]
        for name in names
    ]

    x = list(
        range(len(names))
    )

    fig, ax1 = plt.subplots(
        figsize=(10, 6),
    )

    bars = ax1.bar(
        x,
        invalid_rates,
        width=0.45,
        label="Invalid Action Rate",
    )

    ax1.set_ylabel(
        "Invalid Action Rate (%)"
    )

    ax1.set_xticks(
        x
    )

    ax1.set_xticklabels(
        names,
        rotation=10,
    )

    ax1.grid(
        axis="y",
        alpha=0.3,
    )

    ax2 = ax1.twinx()

    ax2.plot(
        x,
        planner_steps,
        marker="o",
        linewidth=2,
        label="Average Planner Steps",
    )

    ax2.set_ylabel(
        "Average Planner Steps"
    )

    ax1.set_title(
        "Planner and ActionGuard Metrics"
    )

    ax1.bar_label(
        bars,
        labels=[
            f"{value:.1f}%"
            for value in invalid_rates
        ],
        padding=3,
    )

    for index, value in enumerate(planner_steps):
        ax2.text(
            index,
            value - 0.015,
            f"{value:.2f}",
            ha="center",
            va="top",
        )

    lines1, labels1 = (
        ax1.get_legend_handles_labels()
    )

    lines2, labels2 = (
        ax2.get_legend_handles_labels()
    )

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "04_planner_metrics.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ==================== 主程序 ====================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = load_all_results()

    main_result = results[
        MAIN_RESULT
    ]

    plot_success_rates(
        main_result
    )

    plot_error_metrics(
        main_result
    )

    plot_benchmark_comparison(
        results
    )

    plot_planner_metrics(
        results
    )

    print(
        "\nResults visualization completed."
    )

    print(
        "Figures saved to:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()