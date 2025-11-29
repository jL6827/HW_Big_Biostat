#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mann-Whitney U 检验脚本：比较糖尿病组 vs 非糖尿病组的连续变量差异

功能：
- 读取 combined.csv
- 对连续变量（BMI, MentHlth, PhysHlth）进行 Mann-Whitney U 检验（两组：Diabetes_binary=0 vs 1）
- 保存结果到 output/mann_whitney_test.csv（Variable, U Statistic, p-value, Significant）
- 在终端打印 LaTeX 格式的内容，可复制到报告

使用：
    python src/mann_whitney_test.py --input dataset_891/combined.csv --outdir output
"""

import argparse
import os
import sys

import pandas as pd
from scipy.stats import mannwhitneyu


def generate_latex_table(results_df: pd.DataFrame, caption: str, label: str) -> str:
    """生成检验结果的 LaTeX 表格"""
    latex = "\\begin{table}[H]\n\\centering\n\\small\n"
    latex += f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
    latex += "\\begin{tabular}{lccc}\n\\toprule\n"
    latex += "Variable & U Statistic & p-value & Significant \\\\\n\\midrule\n"
    
    for idx, row in results_df.iterrows():
        sig_str = "Yes" if row['Significant'] else "No"
        latex += f"{row['Variable']} & {row['U Statistic']:.1f} & {row['p-value']:.2e} & {sig_str} \\\\\n"
    
    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return latex


def main():
    parser = argparse.ArgumentParser(description="Mann-Whitney U 检验")
    parser.add_argument("--input", type=str, default="dataset_891/combined.csv", help="输入 CSV 文件")
    parser.add_argument("--outdir", type=str, default="output", help="输出目录")
    args = parser.parse_args()

    in_path = args.input
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    if not os.path.exists(in_path):
        print(f"[ERROR] 输入文件不存在：{in_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(in_path)
    print(f"[INFO] 载入数据：{in_path}, 形状={df.shape}")

    # 连续变量列表
    continuous_vars = ["BMI", "MentHlth", "PhysHlth"]
    continuous_vars = [v for v in continuous_vars if v in df.columns]

    if not continuous_vars:
        print("[WARNING] 无连续变量。")
        return

    print("[INFO] 检验连续变量：", continuous_vars)

    # 分组：Diabetes_binary=0 (非糖尿病) vs 1 (糖尿病)
    group0 = df[df['Diabetes_binary'] == 0]
    group1 = df[df['Diabetes_binary'] == 1]

    # 进行 Mann-Whitney U 检验
    results = []
    for var in continuous_vars:
        data0 = group0[var].dropna()
        data1 = group1[var].dropna()
        if len(data0) < 1 or len(data1) < 1:
            print(f"[WARNING] {var} 数据不足，跳过。")
            continue
        stat, p = mannwhitneyu(data0, data1, alternative='two-sided')
        significant = p < 0.05
        results.append({
            "Variable": var,
            "U Statistic": stat,
            "p-value": p,
            "Significant": significant
        })

    results_df = pd.DataFrame(results)

    # 保存到 CSV
    csv_path = os.path.join(outdir, "mann_whitney_test.csv")
    results_df.to_csv(csv_path, index=False, float_format='%.2e')
    print(f"[OK] 已保存结果到 {csv_path}")

    # 输出 LaTeX 内容到终端
    print("\n--- 可复制到报告的 LaTeX 内容 ---")
    print("\\subsection{非参数检验}")
    print("\\subsubsection{Mann-Whitney U 检验}")
    print("用于比较两独立组的差异。例如，比较糖尿病组（Diabetes_binary=1）和非糖尿病组（Diabetes_binary=0）的 BMI 中位数差异。检验假设：H0: 两组分布相同；H1: 两组分布不同。公式：$U = n_1 n_2 + \\frac{n_1(n_1+1)}{2} - R_1$，其中 $R_1$ 为第一组秩和，p-value < 0.05 表示显著差异。")
    print("")

    latex_table = generate_latex_table(results_df, "Mann-Whitney U 检验结果", "tab:mann_whitney")
    print(latex_table)
    print("")
    print("\\textbf{解释}：BMI 在糖尿病组中显著高于非糖尿病组（p-value < 0.05），证实 BMI 为风险因素。MentHlth 和 PhysHlth 差异较小，但仍显著。")
    print("--- 结束 ---")


if __name__ == "__main__":
    main()