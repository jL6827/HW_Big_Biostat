#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据探索脚本：展示数据的基本信息和结构

功能：
- 读取 combined.csv
- 在终端打印数据的基本信息和结构（LaTeX 格式，可复制到报告）

使用：
    python src/data_exploration.py --input dataset_891/combined.csv
"""

import argparse
import sys
import os
import pandas as pd


def generate_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """生成描述性统计的 LaTeX 表格字符串"""
    desc = df.describe().transpose()
    desc = desc.round(2)  # 保留2位小数
    
    latex = "\\begin{table}[H]\n\\centering\n"
    latex += f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
    latex += "\\begin{tabular}{l" + "c" * len(desc.columns) + "}\n\\toprule\n"
    
    # 表头
    header = "Variable & " + " & ".join(desc.columns) + " \\\\\n\\midrule\n"
    latex += header
    
    # 数据行
    for idx, row in desc.iterrows():
        row_str = f"{idx} & " + " & ".join(f"{val:.2f}" for val in row.values) + " \\\\\n"
        latex += row_str
    
    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return latex


def main():
    parser = argparse.ArgumentParser(description="数据探索：基本信息和结构")
    parser.add_argument("--input", type=str, default="dataset_891/combined.csv", help="输入 CSV 文件")
    args = parser.parse_args()

    in_path = args.input

    if not os.path.exists(in_path):
        print(f"[ERROR] 输入文件不存在：{in_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(in_path)
    print(f"[INFO] 载入数据：{in_path}, 形状={df.shape}")

    # 输出 LaTeX 内容到终端
    print("\\subsection{数据探索}")
    print("展示数据的基本信息和结构。")
    print("")

    # 数据形状
    print(f"数据集包含 {df.shape[0]} 行和 {df.shape[1]} 列。")
    print("")

    # 列名和数据类型
    print("\\begin{itemize}")
    for col in df.columns:
        dtype = str(df[col].dtype)
        print(f"\\item {col}: {dtype}")
    print("\\end{itemize}")
    print("")

    # 描述性统计表格（仅数值列）
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        desc_df = df[numeric_cols].describe()
        latex_table = generate_latex_table(desc_df, "数值变量描述性统计", "tab:desc_stats")
        print(latex_table)
        print("")
    else:
        print("无数值变量。")
        print("")

    # 缺失值检查
    missing = df.isnull().sum()
    total_missing = missing.sum()
    print(f"数据集总缺失值数量：{total_missing}。")
    print("")
    if total_missing > 0:
        print("\\begin{itemize}")
        for col, count in missing[missing > 0].items():
            pct = (count / len(df)) * 100
            print(f"\\item {col}: {count} ({pct:.2f}\\%))")
        print("\\end{itemize}")
        print("")

    # 重复行检查
    dup_count = df.duplicated().sum()
    print(f"数据集重复行数量：{dup_count}。")
    print("")

    # 目标变量分布（如果存在）
    target_col = "Diabetes_binary"
    if target_col in df.columns:
        dist = df[target_col].value_counts().sort_index()
        print(f"目标变量 {target_col} 分布：")
        print("\\begin{itemize}")
        for val, count in dist.items():
            pct = (count / len(df)) * 100
            print(f"\\item {val}: {count} ({pct:.2f}\\%))")
        print("\\end{itemize}")
        print("")


if __name__ == "__main__":
    main()