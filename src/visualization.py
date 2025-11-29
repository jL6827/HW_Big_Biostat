#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化分析脚本：单变量和多变量图表

功能：
- 读取 combined.csv
- 生成单变量图表（直方图、箱线图、条形图）
- 生成多变量图表（热力图、堆叠条形图、箱线图）
- 保存高质量图表到 figs/（子文件夹区分）
- 终端打印图表文件名列表

使用：
    python src/visualization.py --input dataset_891/combined.csv --figs_dir figs
"""

import argparse
import os
import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 设置 seaborn 主题：好看风格
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

def plot_univariate(df: pd.DataFrame, var: str, plot_type: str, save_path: str):
    """生成单变量图表"""
    plt.figure(figsize=(8, 6))
    
    if plot_type == "hist":
        # 直方图 + KDE
        sns.histplot(df[var], kde=True, bins=30, color="skyblue")
        plt.title(f"{var} Distribution Histogram")
        plt.xlabel(var)
        plt.ylabel("Frequency")
    elif plot_type == "box":
        # 箱线图
        sns.boxplot(y=df[var], color="lightgreen")
        plt.title(f"{var} Box Plot")
        plt.ylabel(var)
    elif plot_type == "bar":
        # 分类条形图
        sns.countplot(x=df[var], palette="pastel")
        plt.title(f"{var} Frequency Bar Plot")
        plt.xlabel(var)
        plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"已保存单变量图：{save_path}")

def plot_multivariate(df: pd.DataFrame, vars_list: list, plot_type: str, save_path: str):
    """生成多变量图表"""
    plt.figure(figsize=(10, 8))
    
    if plot_type == "heatmap":
        # 相关性热力图
        corr = df[vars_list].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
        plt.title("Numerical Variables Correlation Heatmap")
    elif plot_type == "stacked_bar":
        # 堆叠条形图（分类 vs 目标）
        var, target = vars_list[0], vars_list[1]
        cross_tab = pd.crosstab(df[var], df[target], normalize="index") * 100
        cross_tab.plot(kind="bar", stacked=True, colormap="viridis")
        plt.title(f"{var} vs {target} Stacked Bar Plot (Percentage)")
        plt.ylabel("Percentage (%)")
        plt.legend(title=target)
    elif plot_type == "box_hue":
        # 数值 vs 分类箱线图
        num_var, cat_var = vars_list[0], vars_list[1]
        sns.boxplot(x=df[cat_var], y=df[num_var], palette="Set2")
        plt.title(f"{num_var} vs {cat_var} Box Plot")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"已保存多变量图：{save_path}")

def main():
    parser = argparse.ArgumentParser(description="可视化分析")
    parser.add_argument("--input", type=str, default="dataset_891/combined.csv", help="输入 CSV 文件")
    parser.add_argument("--figs_dir", type=str, default="figs", help="图表输出目录")
    args = parser.parse_args()

    in_path = args.input
    figs_dir = args.figs_dir
    os.makedirs(os.path.join(figs_dir, "univariate"), exist_ok=True)
    os.makedirs(os.path.join(figs_dir, "multivariate"), exist_ok=True)

    if not os.path.exists(in_path):
        print(f"[ERROR] 输入文件不存在：{in_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(in_path)
    print(f"[INFO] 载入数据：{in_path}, 形状={df.shape}")

    # 单变量分析：选取变量
    univariate_vars = {
        "BMI": "hist",  # 直方图
        "MentHlth": "box",  # 箱线图
        "PhysHlth": "hist",  # 直方图
        "HighBP": "bar",  # 条形图
        "Smoker": "bar",  # 条形图
    }
    for var, plot_type in univariate_vars.items():
        if var in df.columns:
            save_path = os.path.join(figs_dir, "univariate", f"{var}_{plot_type}.png")
            plot_univariate(df, var, plot_type, save_path)

    # 多变量分析：选取组合（移除 pairplot）
    multivariate_plots = [
        (["BMI", "MentHlth", "PhysHlth"], "heatmap"),  # 热力图
        (["HighBP", "Diabetes_binary"], "stacked_bar"),  # 堆叠条形图
        (["Smoker", "Diabetes_binary"], "stacked_bar"),
        (["BMI", "Diabetes_binary"], "box_hue"),  # 箱线图（数值 vs 分类）
    ]
    for vars_list, plot_type in multivariate_plots:
        if all(v in df.columns for v in vars_list):
            save_path = os.path.join(figs_dir, "multivariate", f"{'_'.join(vars_list)}_{plot_type}.png")
            plot_multivariate(df, vars_list, plot_type, save_path)

    print("\n[INFO] 图表生成完成。检查 figs/univariate/ 和 figs/multivariate/")

if __name__ == "__main__":
    main()