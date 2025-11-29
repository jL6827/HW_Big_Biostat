#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数值型变量统计脚本：集中趋势、离散程度和分布形态

功能：
- 读取 combined.csv
- 计算数值型变量的：均值、中位数、众数、标准差、方差、极差、偏度、峰度
- 保存统计量到 output/numeric_stats.csv

使用：
    python src/numeric_stats.py --input dataset_891/combined.csv --outdir output
"""

import argparse
import os
import sys
import scipy

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis


def generate_latex_table(stats_df: pd.DataFrame, caption: str, label: str) -> str:
    """生成统计量的 LaTeX 表格字符串"""
    stats_df = stats_df.round(2)  # 保留2位小数
    
    # 使用 \small 缩小字体，并指定列宽（第一列 p{2cm}，其余 c）
    latex = "\\begin{table}[H]\n\\centering\n\\small\n"  # 添加 \small
    latex += f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
    # 第一列设为 p{2cm}（包裹长变量名），其余 c（居中）
    col_spec = "p{2cm}" + "c" * (len(stats_df.columns) - 1)  # 假设第一列是 Variable
    latex += f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n"
    
    # 表头
    header = "Variable & " + " & ".join(stats_df.columns[1:]) + " \\\\\n\\midrule\n"  # 跳过 index 列
    latex += header
    
    # 数据行
    for idx, row in stats_df.iterrows():
        row_str = f"{idx} & " + " & ".join(str(val) for val in row.values[1:]) + " \\\\\n"  # 跳过 index
        latex += row_str
    
    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return latex


def compute_stats(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """计算指定列的统计量"""
    stats = []
    for col in cols:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        mean_val = s.mean()
        median_val = s.median()
        mode_val = s.mode().iloc[0] if not s.mode().empty else np.nan  # 众数（取第一个）
        std_val = s.std()
        var_val = s.var()
        range_val = s.max() - s.min()
        skew_val = skew(s)
        kurt_val = kurtosis(s)  # 峰度
        
        stats.append({
            "Variable": col,
            "Mean": mean_val,
            "Median": median_val,
            "Mode": mode_val,
            "Std": std_val,
            "Var": var_val,
            "Range": range_val,
            "Skew": skew_val,
            "Kurt": kurt_val
        })
    return pd.DataFrame(stats).set_index("Variable")


def main():
    parser = argparse.ArgumentParser(description="数值型变量统计")
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

    # 选择数值型变量（排除目标列，如果不想统计）
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    # 可选：排除目标列
    target_col = "Diabetes_binary"
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    if not numeric_cols:
        print("[WARNING] 无数值型变量。")
        return

    print("[INFO] 数值型变量：", numeric_cols)

    # 计算统计量
    stats_df = compute_stats(df, numeric_cols)

    # 保存到 CSV
    csv_path = os.path.join(outdir, "numeric_stats.csv")
    stats_df.to_csv(csv_path)
    print(f"[OK] 已保存统计量到 {csv_path}")

    # 输出 LaTeX 内容到终端
    print("\n--- 可复制到报告的 LaTeX 内容 ---")
    print("\\subsection{数值型变量统计}")
    print("展示数值型变量的集中趋势、离散程度和分布形态。")
    print("")
    print("\\textbf{主要统计量：}")
    print("\\begin{itemize}")
    print("    \\item 均值、中位数、众数")
    print("    \\item 标准差、方差、极差")
    print("    \\item 偏度、峰度")
    print("\\end{itemize}")
    print("")

    # 生成并打印表格
    latex_table = generate_latex_table(stats_df, "数值型变量主要统计量", "tab:numeric_stats")
    print(latex_table)
    print("--- 结束 ---")


if __name__ == "__main__":
    main()