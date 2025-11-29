#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
正态性检验脚本：对连续型数据进行 Shapiro-Wilk 检验

功能：
- 读取 combined.csv
- 对连续变量（BMI, MentHlth, PhysHlth）进行正态性检验
- 保存结果到 output/normality_test.csv（Variable, Statistic, p-value, Normality）
- 在终端打印 LaTeX 格式的内容，可复制到报告

使用：
    python src/normality_test.py --input dataset_891/combined.csv --outdir output
"""

import argparse
import os
import sys

import pandas as pd
from scipy.stats import shapiro


def generate_latex_table(results_df: pd.DataFrame, caption: str, label: str) -> str:
    """生成检验结果的 LaTeX 表格"""
    latex = "\\begin{table}[H]\n\\centering\n\\small\n"
    latex += f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
    latex += "\\begin{tabular}{lccc}\n\\toprule\n"
    latex += "Variable & Statistic & p-value & Normality \\\\\n\\midrule\n"
    
    for idx, row in results_df.iterrows():
        norm_str = "Yes" if row['Normality'] else "No"
        latex += f"{row['Variable']} & {row['Statistic']:.4f} & {row['p-value']:.2e} & {norm_str} \\\\\n"
    
    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return latex


def main():
    parser = argparse.ArgumentParser(description="正态性检验")
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

    # 进行 Shapiro-Wilk 检验
    results = []
    for var in continuous_vars:
        data = df[var].dropna()
        if len(data) < 3:
            print(f"[WARNING] {var} 数据不足，跳过。")
            continue
        stat, p = shapiro(data)
        normality = p > 0.05  # p > 0.05 表示正态
        results.append({
            "Variable": var,
            "Statistic": stat,
            "p-value": p,
            "Normality": normality
        })

    results_df = pd.DataFrame(results)

    # 保存到 CSV（修改：用 float_format 科学计数法）
    csv_path = os.path.join(outdir, "normality_test.csv")
    results_df.to_csv(csv_path, index=False, float_format='%.2e')
    print(f"[OK] 已保存结果到 {csv_path}")

    # 输出 LaTeX 内容到终端
    print("\n--- 可复制到报告的 LaTeX 内容 ---")
    print("\\section{假设检验}")
    print("\\subsection{正态性检验}")
    print("检验数据是否符合正态分布。对连续型数据进行检验。")
    print("")
    print("使用 Shapiro-Wilk 检验，p-value > 0.05 表示数据符合正态分布。")
    print("")

    latex_table = generate_latex_table(results_df, "连续变量正态性检验结果", "tab:normality_test")
    print(latex_table)
    print("--- 结束 ---")


if __name__ == "__main__":
    main()