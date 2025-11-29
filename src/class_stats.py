#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分类变量统计脚本：频数分布和比例

功能：
- 读取 combined.csv
- 识别分类变量（非数值列 + 数值列但唯一值 <=10）
- 计算每个分类变量的频数和比例
- 保存结果到 output/class_stats.csv（格式：Variable, Category, Count, Percentage）
- 在终端打印 LaTeX 格式的内容，可复制到报告

使用：
    python src/class_stats.py --input dataset_891/combined.csv --outdir output
"""

import argparse
import os
import sys

import pandas as pd


def generate_latex_for_variable(var_name: str, counts: pd.Series) -> str:
    """为单个变量生成 LaTeX 表格"""
    # 过滤掉 NaN
    counts = counts.dropna()
    
    latex = f"\\textbf{{{var_name}}}\n"
    latex += "\\begin{table}[H]\n\\centering\n\\small\n"
    latex += f"\\caption{{{var_name} 频数分布}}\n"
    latex += "\\begin{tabular}{lcc}\n\\toprule\n"
    latex += "类别 & 频数 & 比例 (\%) \\\\\n\\midrule\n"
    
    total = counts.sum()
    for val, cnt in counts.items():
        pct = (cnt / total) * 100
        latex += f"{val} & {cnt} & {pct:.2f} \\\\\n"
    
    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n\n"
    return latex


def main():
    parser = argparse.ArgumentParser(description="分类变量统计")
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

    # 识别分类变量：非数值列 + 数值列但唯一值 <=10（假设是类别编码）
    categorical_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            categorical_cols.append(col)
        elif df[col].dtype in ['int64', 'float64']:
            if df[col].nunique() <= 10:  # 假设 <=10 个唯一值的是分类
                categorical_cols.append(col)
    
    # 排除目标列（如果不想统计）
    target_col = "Diabetes_binary"
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    if not categorical_cols:
        print("[WARNING] 无分类变量。")
        return

    print("[INFO] 分类变量：", categorical_cols)

    # 收集所有数据用于 CSV
    csv_data = []
    for col in categorical_cols:
        counts = df[col].value_counts().dropna()
        total = counts.sum()
        for val, cnt in counts.items():
            pct = (cnt / total) * 100
            csv_data.append({
                "Variable": col,
                "Category": val,
                "Count": cnt,
                "Percentage": round(pct, 2)
            })
    
    csv_df = pd.DataFrame(csv_data)
    csv_path = os.path.join(outdir, "class_stats.csv")
    csv_df.to_csv(csv_path, index=False)
    print(f"[OK] 已保存结果到 {csv_path}")

    # 输出 LaTeX 内容到终端
    print("\n--- 可复制到报告的 LaTeX 内容 ---")
    print("\\subsection{分类变量统计}")
    print("展示分类变量的频数分布和比例。")
    print("")
    for col in categorical_cols:
        counts = df[col].value_counts()
        latex_content = generate_latex_for_variable(col, counts)
        print(latex_content)
    print("--- 结束 ---")


if __name__ == "__main__":
    main()